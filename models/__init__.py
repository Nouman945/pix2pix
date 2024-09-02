import importlib
import sys  # Import sys module here
from models.base_model import BaseModel

def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py"."""
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(f"In {model_filename}.py, there should be a subclass of BaseModel with class name that matches {target_model_name} in lowercase.")
        sys.exit(0)  # Use sys.exit(0) instead of exit(0)

    return model

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    """Create a model given the option."""
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model [{type(instance).__name__}] was created")
    return instance
