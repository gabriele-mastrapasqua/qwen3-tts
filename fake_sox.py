# Fake sox module to bypass import error
class Transforms:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return args[0] if args else None

def build(transforms, *args, **kwargs):
    return Transforms()
