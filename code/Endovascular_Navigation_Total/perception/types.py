GUIDEWIRE_NAMES = {"guidewire", "wire", "magnetic_guidewire", "magnetic-guidewire"}
PARTICLE_NAMES = {"particle", "sphere", "magnetic_particle", "magnetic-particle", "sphere_mag"}


def normalize_target_name(name):
    if name is None:
        return "unknown"
    key = str(name).strip().lower()
    if key in GUIDEWIRE_NAMES:
        return "guidewire"
    if key in PARTICLE_NAMES:
        return "particle"
    return key


def target_matches(class_name, target_type):
    normalized = normalize_target_name(class_name)
    if target_type in (None, "both", "all"):
        return normalized in {"guidewire", "particle"}
    return normalized == normalize_target_name(target_type)

