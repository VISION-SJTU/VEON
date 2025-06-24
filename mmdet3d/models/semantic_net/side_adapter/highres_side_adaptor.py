import torch
from torch import nn
from typing import List, Dict

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from detectron2.config import configurable
from torch.utils.checkpoint import checkpoint
from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

HIGHRES_SIDE_ADAPTOR_REGISTRY = Registry("HIGHRES_SIDE_ADAPTOR")
HIGHRES_SIDE_ADAPTOR_REGISTRY.__doc__ = """
Registry for high resolution side adaptor.
"""

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=-1):
        super().__init__()
        out_dim = dim if out_dim == -1 else out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=-1):
        super().__init__()
        out_dim = dim if out_dim == -1 else out_dim
        self.conv1 = nn.Conv2d(dim, hidden_dim, stride=1, padding=1, kernel_size=3)
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, stride=1, padding=1, kernel_size=3)
        self.ln2 = nn.LayerNorm(out_dim)
        self.dim, self.h_dim, self.out_dim = dim, hidden_dim, out_dim

    def forward(self, x, size=(1, 1)):
        B, L, dim = x.shape
        H, W = size
        assert H * W == L
        x = x.permute(0, 2, 1).reshape(B, dim, H, W).contiguous()
        x = self.gelu(self.conv1(x))
        x = self.ln1(x.reshape(B, self.h_dim, L).permute(0, 2, 1))
        x = x.permute(0, 2, 1).reshape(B, self.h_dim, H, W).contiguous()
        x = self.conv2(x)
        x = self.ln2(x.reshape(B, self.out_dim, L).permute(0, 2, 1))
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x, x_pos):
        x = self.norm(x)
        q, k, v = self.to_q(x + x_pos), self.to_k(x + x_pos), self.to_v(x)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x, x_pos, ext, ext_pos):
        x = self.norm(x)
        q, k, v = self.to_q(x + x_pos), self.to_k(ext + ext_pos), self.to_v(ext)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class HighresSideAdaptorBlock(nn.Module):
    def __init__(self, dim, mlp_dim=960,
                 neck_dim=0, pre_norm=False, use_add=False, use_checkpoint=False):
        super(HighresSideAdaptorBlock, self).__init__()
        self.ff = ConvBlock(dim, mlp_dim)
        self.use_checkpoint = use_checkpoint
        self.neck_add = nn.Linear(neck_dim, dim, bias=False) if neck_dim > 0 and use_add else nn.Identity()
        self.use_add = use_add
        self.pre_norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.ln_3 = nn.LayerNorm(dim)
        self.ln_4 = nn.LayerNorm(dim)

    def forward(self, x, x_pos, ext, ext_pos, offset=None, offset_shape=(1, 1)):
        B, C_clip, h_ext, w_ext = ext.shape
        x = self.pre_norm(x)
        if self.use_checkpoint:
            x = checkpoint(self.ff, self.ln_3(x), offset_shape) + x
        else:
            x = self.ff(self.ln_3(x), offset_shape) + x
        if offset is not None:
            offset = self.neck_add(offset.reshape(B, C_clip, -1).permute(0, 2, 1))
            offset = nn.functional.interpolate(offset.permute(0, 2, 1).reshape(B, -1, h_ext, w_ext), size=offset_shape)
            offset = offset.reshape(B, offset.shape[1], -1).permute(0, 2, 1)
            x[:, -offset.shape[1]:, :] = x[:, -offset.shape[1]:, :] + offset

        return self.ln_4(x)


class AttnManipulateBlock(nn.Module):
    def __init__(self, dim, mlp_dim=768, clip_dim=1024, heads=16, dim_head=64, attn_layers=6,
                 add_layers=2, supp_dim=384, pre_norm=False, use_checkpoint=False):
        super(AttnManipulateBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.pre_norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.ff = ConvBlock(dim, mlp_dim, mlp_dim)

        # self.head_offset = FeedForward(mlp_dim, mlp_dim, clip_dim * add_layers)
        self.dim, self.mlp_dim, self.clip_dim = dim, mlp_dim, clip_dim
        self.add_layers, self.attn_layers, self.heads, self.dim_head = add_layers, attn_layers, heads, dim_head
        self.attn_out = attn_layers * heads * dim_head

        self.head_attn = FeedForward(mlp_dim, mlp_dim, self.attn_out)
        self.head_supp = FeedForward(mlp_dim, mlp_dim, supp_dim)

        self.ln_3 = nn.LayerNorm(dim)
        self.ln_4 = nn.LayerNorm(mlp_dim)

    def forward(self, x, side_shape=(1, 1), new_shape=(1, 1)):
        x = self.pre_norm(x)
        if self.use_checkpoint:
            x = checkpoint(self.ff, self.ln_3(x), side_shape)
            x = self.ln_4(x)
            # offsets = checkpoint(self.head_offset, x)
            attns = checkpoint(self.head_attn, x)
            supp = checkpoint(self.head_supp, x)
        else:
            x = self.ff(self.ln_3(x), side_shape)
            x = self.ln_4(x)
            # offsets = self.head_offset(x)
            attns = self.head_attn(x)
            supp = self.head_supp(x)

        H, W = side_shape
        h, w = new_shape
        B = x.shape[0]

        # offsets = offsets.permute(0, 2, 1).reshape(B, -1, H, W)
        # offsets = nn.functional.interpolate(offsets, size=(h, w), mode="bilinear")
        # offsets = offsets.reshape(B, h * w, self.add_layers, self.clip_dim).permute(2, 0, 1, 3)

        attns = attns.permute(0, 2, 1).reshape(B, -1, H, W)
        attns = nn.functional.interpolate(attns, size=(h, w), mode="bilinear").reshape(B, h, w, -1)
        attns = attns.reshape(B, h * w, self.attn_layers, self.heads, self.dim_head)
        attns = torch.einsum("bmahd,bnahd->bmnah", attns, attns).permute(3, 0, 4, 1, 2)

        supp = supp.permute(0, 2, 1).reshape(B, -1, H, W)

        # return offsets, attns, supp

        return None, attns, supp


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=True,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        _, c, h, w = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, (h, w)


@HIGHRES_SIDE_ADAPTOR_REGISTRY.register()
class HighresSideAdaptorNetwork(nn.Module):
    @configurable
    def __init__(self, patch_embed, hsa_net_body, rear_block, cr_map, use_checkpoint=False):
        super().__init__()

        self.patch_embed = patch_embed
        self.hsa_net_body = hsa_net_body
        self.rear_block = rear_block
        self.cr_map = cr_map
        self.use_checkpoint = use_checkpoint

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):

        # Parameters
        dim_features = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.DIM
        dim_clip = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.CLIP_DIM
        dim_mlp = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.MLP_DIM
        input_shape = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.INPUT_SIZE
        patch_shape = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.PATCH_SIZE
        num_heads = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.NUM_HEADS
        
        # Layers of HSA fusion
        cross_attn_map: List[str] = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.FUSION_MAP
        cr_map = {int(i): (int(j), int(k)) for i, j, k in [x.split("->") for x in cross_attn_map]}

        # Network structure
        patch_shape = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.PATCH_SHAPE
        patch_embed = PatchEmbed(input_shape, patch_shape, embed_dim=dim_features, norm_layer=False)

        hsa_net_body = nn.ModuleList([])
        for i in range(len(cross_attn_map)):
            hsa_block = HighresSideAdaptorBlock(dim=dim_features, neck_dim=dim_clip, mlp_dim=dim_mlp,
                                            pre_norm=(i == 0), use_add=cr_map[i][1] >= 0, use_checkpoint=True)
            hsa_net_body.append(hsa_block)
        
        manip_dim_head = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.ATTN_MANIP.DIM_HEAD
        manip_attn_layers = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.ATTN_MANIP.ATTN_LAYERS
        manip_add_layers = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.ATTN_MANIP.ADD_LAYERS
        manip_supp_dim = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.ATTN_MANIP.SUPP_DIM

        rear_block = AttnManipulateBlock(dim=dim_features, mlp_dim=dim_mlp, clip_dim=dim_clip,
                                          heads=num_heads, dim_head=manip_dim_head, attn_layers=manip_attn_layers,
                                          add_layers=manip_add_layers, supp_dim=manip_supp_dim, 
                                          pre_norm=False, use_checkpoint=True)

        return {
            "patch_embed": patch_embed,
            "hsa_net_body": hsa_net_body,
            "rear_block": rear_block,
            "cr_map": cr_map,
        }


    def forward(self, image: torch.Tensor, clip_features: List):

        x, (H, W) = self.patch_embed(image)
        B, h, w = x.shape[0], clip_features[1].shape[2], clip_features[1].shape[3]
        for layer_id, hsa_block in enumerate(self.hsa_net_body):
            ca_id, add_id = self.cr_map[layer_id]
            if self.use_checkpoint:
                x = checkpoint(hsa_block, x, None, clip_features[ca_id].contiguous(), None,
                             clip_features[add_id].contiguous() if hsa_block.use_add else None, (H, W))
            else:
                x = hsa_block(x, None, clip_features[ca_id].contiguous(), None,
                             clip_features[add_id].contiguous() if hsa_block.use_add else None, (H, W))
        if self.use_checkpoint:
            offsets, attns, supp = checkpoint(self.rear_block, x, (H, W), (h, w))
        else:
            offsets, attns, supp = self.rear_block(x, (H, W), (h, w))

        return offsets, attns, supp



def build_hsa_network(cfg, input_shape):
    name = cfg.MODEL.HIGHRES_SIDE_ADAPTOR.NAME
    return HIGHRES_SIDE_ADAPTOR_REGISTRY.get(name)(cfg, input_shape)
