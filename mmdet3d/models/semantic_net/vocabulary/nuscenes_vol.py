NUSCENES_CLASSES_BRIEF = [
    {"category": "others", "detailed_items": [
        ("debris",),
        ("animal", ),
        ("personal mobility", ),
        ("skateboard", ),
        ("segway", ),
        ("scooter", ),
        ("stroller", ),
        ("wheelchair", ),
        ("trash bag", ),
        ("road sign", ),
        ("trash can", ),
        ("wheel barrow", ),
        ("garbage-bin with wheels", ),
        # ("shopping cart", ),
        ("bicycle rack", ),
        ("ambulance vehicle", ),
        ("police vehicle", ),
        # ("noise", ),
        # ("dust", ),
        # ("vapor" ,),
        # ("fog", ),
        # ("raindrop", ),
        # ("smoke", ),
        # ("reflection", ),
        # ("ego vehicle", ),
    ]},
    {"category": "barrier", "detailed_items": [
        ("traffic barrier", )
    ]},
    {"category": "bicycle", "detailed_items": [
        ("bicycle", )
    ]},
    {"category": "bus", "detailed_items": [
        ("bus", )
    ]},
    {"category": "car", "detailed_items": [
        ("car", ),
        ("sedan", ),
        ("hatch-back", ),
        ("wagon", ),
        ("van", ),
        ("mini-van",),
        ("SUV", ),
        ("jeep", )
    ]},
    {"category": "construction_vehicle", "detailed_items": [
        ("construction vehicle", )
    ]},
    {"category": "motorcycle", "detailed_items": [
        ("motorcycle", ),
        # ("3-wheel vehicle", )
    ]},
    {"category": "pedestrian", "detailed_items": [
        ("pedestrian", ),
        ("construction worker", ),
        ("police officer", )
    ]},
    {"category": "traffic_cone", "detailed_items": [
        ("traffic cone", )
    ]},
    {"category": "trailer", "detailed_items": [
        ("trailer", )
    ]},
    {"category": "truck", "detailed_items": [
        ("truck", ),
        # ("semi-tractor", ),
        # ("lorry", )
    ]},
    {"category": "driveable surface", "detailed_items": [
        ("road", )
    ]},
    {"category": "other flat", "detailed_items": [
        ("traffic delimiter", ),
        ("traffic island",),
        ("rail track", ),
        ("lake", ),
        ("river", ),
    ]},
    {"category": "sidewalk", "detailed_items": [
        ("sidewalk", ),
        ("pedestrian walkway", ),
        ("bike path", ),
    ]},
    {"category": "terrain", "detailed_items": [
        # ("terrain", ),
        ("grass", ),
        ("rolling hill", ),
        ("soil", ),
        ("sand", ),
        ("gravel", ),
    ]},
    {"category": "manmade", "detailed_items": [
        ("building", ),
        ("wall", ),
        ("guard rail", ),
        ("fence", ),
        # ("pole", ),
        ("drainage", ),
        ("hydrant", ),
        ("flag", ),
        ("banner", ),
        ("street sign", ),
        ("electric circuit box", ),
        ("traffic light", ),
        ("parking meter", ),
        ("stairs", ),
    ]},
    {"category": "vegetation", "detailed_items": [
        ("vegetation", ),
        ("plants",),
        ("bushes", ),
        # ("potted plant", ),
        ("tree", ),
    ]},
]


NUSCENES_CLASSES = [
    {"category": "others", "detailed_items": [
        ("animal", "All animals, e.g. cats, rats, dogs, deer, birds."),
        ("personal mobility", "A small electric or self-propelled vehicle, e.g. skateboard, segway, or scooters, on which the person typically travels in a upright position."),
        ("stroller", "Any stroller."),
        ("wheelchair", "Any type of wheelchair."),
        ("debris", "Debris or movable object that is too large to be driven over safely. Includes misc. things like trash bags, temporary road-signs, objects around construction zones, and trash cans."),
        ("pushable pullable objects", "Objects that a pedestrian may push or pull. For example dolleys, wheel barrows, garbage-bins with wheels, or shopping carts. Typically not designed to carry humans."),
        ("bicycle rack", "Area or device intended to park or secure the bicycles in a row. It includes all the bicycles parked in it and any empty slots that are intended for parking bicycles. Bicycles that are not part of the rack should not be included."),
        ("ambulance vehicle", "All types of ambulances."),
        ("police vehicle", "All types of police vehicles including police bicycles and motorcycles."),
        # ("noise", "Things that does not correspond to a physical object, such as dust, vapor, noise, fog, raindrops, smoke and reflections."),
        # ("other static", "Objects in the background that are not distinguishable. Or objects that do not match any of the above labels."),
        ("ego vehicle", "The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image."),
    ]},
    {"category": "barrier", "detailed_items": [
        ("traffic barrier", "Any metal, concrete or water barrier temporarily placed in the scene in order to re-direct vehicle or pedestrian traffic. In particular, includes barriers used at construction zones.")
    ]},
    {"category": "bicycle", "detailed_items": [
        ("bicycle", "Human or electric powered 2-wheeled vehicle designed to travel at lower speeds either on road surface, sidewalks or bicycle paths.")
    ]},
    {"category": "bus", "detailed_items": [
        ("bus", "Any types of buses and shuttles designed to carry more than 10 people.")
    ]},
    {"category": "car", "detailed_items": [
        ("car", "Vehicle designed primarily for personal use, e.g. sedans, hatch-backs, wagons, vans, mini-vans, SUVs and jeeps.")
    ]},
    {"category": "construction_vehicle", "detailed_items": [
        ("construction_vehicle", "Vehicles primarily designed for construction. Typically very slow moving or stationary. Cranes and extremities of construction vehicles are only included in annotations if they interfere with traffic."
                                 "Trucks used to hauling rocks or building materials are considered trucks rather than construction vehicles.")
    ]},
    {"category": "motorcycle", "detailed_items": [
        ("motorcycle", "Gasoline or electric powered 2-wheeled vehicle designed to move rapidly (at the speed of standard cars) on the road surface. "
                       "This category includes all motorcycles, vespas and scooters. It also includes light 3-wheel vehicles, often with a light plastic roof and open on the sides, that tend to be common in Asia.")
    ]},
    {"category": "pedestrian", "detailed_items": [
        ("pedestrian", "A pedestrian moving around the cityscape."),
        ("construction worker", "A human in the scene whose main purpose is construction work."),
        ("police_officer", "Any type of police officer, regardless whether directing the traffic or not.")
    ]},
    {"category": "traffic_cone", "detailed_items": [
        ("traffic_cone", "All types of traffic cones.")
    ]},
    {"category": "trailer", "detailed_items": [
        ("trailer", "Any vehicle trailer, both for trucks, cars and motorcycles (regardless of whether currently being towed or not).")
    ]},
    {"category": "truck", "detailed_items": [
        ("truck", "Vehicles primarily designed to haul cargo including pick-ups, lorrys, trucks and semi-tractors.")
    ]},
    {"category": "driveable surface", "detailed_items": [
        ("driveable surface", "All paved or unpaved surfaces that a car can drive on with no concern of traffic rules.")
    ]},
    {"category": "other flat", "detailed_items": [
        ("other flat", "All other forms of horizontal ground-level structures that do not belong to any of driveable surface, curb, sidewalk and terrain. "
                       "Includes elevated parts of traffic islands, delimiters, rail tracks, stairs with at most 3 steps and larger bodies of water (lakes, rivers).")
    ]},
    {"category": "sidewalk", "detailed_items": [
        ("sidewalk", "Sidewalk, pedestrian walkways, bike paths, etc. Part of the ground designated for pedestrians or cyclists. Sidewalks do not have to be next to a road.")
    ]},
    {"category": "terrain", "detailed_items": [
        ("terrain", "Natural horizontal surfaces such as ground level horizontal vegetation (< 20 cm tall), grass, rolling hills, soil, sand and gravel."),
        # ("grass land", "Land where grass or grasslike vegetation grows and is the dominant form of plant life."),
    ]},
    {"category": "manmade", "detailed_items": [
        ("manmade", "Includes man-made structures but not limited to: buildings, walls, guard rails, fences, poles, drainages, hydrants, flags, banners, street signs, electric circuit boxes, traffic lights, parking meters and stairs with more than 3 steps.")
    ]},
    {"category": "vegetation", "detailed_items": [
        ("vegetation", "Any vegetation in the frame that is higher than the ground, including bushes, plants, potted plants, trees, etc. Only tall grass (> 20cm) is part of this"),
        # ("bush", "Dense vegetation consisting of stunted trees or bushes."),
        # ("tree", "Tall perennial woody plant having a main trunk and branches forming a distinct elevated crown.")
    ]},
]
