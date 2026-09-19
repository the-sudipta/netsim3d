"""Vehicle taxonomy built on top of the ImageNet-1k label set.

ResNet/MobileNet trained on ImageNet-1k already separate about 40 vehicle
classes, which is what makes "what kind of car is this?" work without any
extra training. Keys are matched case-insensitively against the label list
that ships with the torchvision weights, so any key that does not exist in a
given model's label set is silently ignored.
"""

# label -> (family, human readable family)
VEHICLE_LABELS = {
    # passenger cars
    "sports car": ("car", "Sports car"),
    "convertible": ("car", "Convertible"),
    "limousine": ("car", "Limousine"),
    "cab": ("car", "Taxi / sedan"),
    "beach wagon": ("car", "Estate / station wagon"),
    "racer": ("car", "Race car"),
    "model t": ("car", "Vintage car"),
    "jeep": ("offroad", "Off-roader"),
    "go-kart": ("kart", "Go-kart"),
    # vans and light commercial
    "minivan": ("van", "Minivan"),
    "moving van": ("van", "Panel van"),
    "police van": ("emergency", "Police van"),
    "recreational vehicle": ("van", "Camper / RV"),
    # trucks
    "pickup": ("truck", "Pickup truck"),
    "tow truck": ("truck", "Tow truck"),
    "trailer truck": ("truck", "Articulated lorry"),
    "garbage truck": ("truck", "Refuse truck"),
    "fire engine": ("emergency", "Fire engine"),
    "ambulance": ("emergency", "Ambulance"),
    # buses
    "school bus": ("bus", "School bus"),
    "minibus": ("bus", "Minibus"),
    "trolleybus": ("bus", "Trolleybus"),
    # two wheelers
    "moped": ("twowheel", "Moped"),
    "motor scooter": ("twowheel", "Scooter"),
    "mountain bike": ("cycle", "Bicycle"),
    "bicycle-built-for-two": ("cycle", "Tandem bicycle"),
    "tricycle": ("cycle", "Tricycle"),
    "unicycle": ("cycle", "Unicycle"),
    # rail
    "freight car": ("rail", "Freight wagon"),
    "passenger car": ("rail", "Rail carriage"),
    "electric locomotive": ("rail", "Electric locomotive"),
    "steam locomotive": ("rail", "Steam locomotive"),
    "streetcar": ("rail", "Tram"),
    "bullet train": ("rail", "High-speed train"),
    # work, farm, military
    "forklift": ("work", "Forklift"),
    "snowplow": ("work", "Snowplough"),
    "golfcart": ("work", "Golf cart"),
    "tractor": ("farm", "Tractor"),
    "harvester": ("farm", "Harvester"),
    "thresher": ("farm", "Thresher"),
    "tank": ("military", "Tank"),
    "half track": ("military", "Half-track"),
    "amphibian": ("military", "Amphibious vehicle"),
}

# Parts, not whole vehicles. Useful as a hint when the crop is too tight.
PART_LABELS = {
    "car wheel": "a wheel",
    "car mirror": "a wing mirror",
    "grille": "a radiator grille",
    "disk brake": "a brake disc",
    "seat belt": "a seat belt",
    "odometer": "a dashboard",
}


FAMILY_PRETTY = {
    "car": "Passenger car",
    "van": "Van",
    "truck": "Truck",
    "bus": "Bus",
    "offroad": "Off-roader",
    "emergency": "Emergency vehicle",
    "kart": "Go-kart",
    "twowheel": "Motorcycle or scooter",
    "cycle": "Bicycle",
    "rail": "Rail vehicle",
    "work": "Work vehicle",
    "farm": "Farm vehicle",
    "military": "Military vehicle",
}


def build_index(categories):
    """Map model class index -> (family, pretty name) for vehicle classes."""
    vehicles, parts = {}, {}
    for idx, name in enumerate(categories):
        key = name.strip().lower()
        if key in VEHICLE_LABELS:
            vehicles[idx] = VEHICLE_LABELS[key]
        elif key in PART_LABELS:
            parts[idx] = PART_LABELS[key]
    return vehicles, parts
