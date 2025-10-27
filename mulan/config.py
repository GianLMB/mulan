"""Definition of Config class"""


from typing import Any, Dict, Optional, Tuple, Union, get_type_hints
from dataclasses import dataclass, asdict
import json


@dataclass
class MulanConfig:
    """
    Configuration class for Mulan.
    Attributes:
        hidden_size (int): The size of the hidden layer (corrsponding to convolution filters).
        last_hidden_size (int): The size of the last hidden layer.
        hidden_dropout_prob (float): The dropout probability for the hidden layer.
        padding_value (float): The padding value.
        kernel_sizes (Union[Tuple, int]): The sizes of the kernels.
        conv_dropout (float): The dropout probability for the convolutional layer.
        add_scores (bool): Whether to add scores or not.
    Methods:
        from_json(cls, json_path, strict=False): Creates an instance of MulanConfig from a JSON file.
        from_dict(cls, args_dict, strict=False): Creates an instance of MulanConfig from a dictionary.
        save(self, json_path=None) -> str: Saves the configuration to a JSON file.
    """
    
    hidden_size: int = 64
    last_hidden_size: int = 20
    hidden_dropout_prob: float = 0.1
    padding_value: float = 0
    kernel_sizes: Union[ Tuple, int] = (1, 5, 9)
    conv_dropout: float = 0.1
    add_scores: bool = False
    
    @classmethod
    def from_json(cls, json_path, strict=False):
        try:
            with open(json_path, "r") as f:
                args_dict = json.load(f)
            return cls.from_dict(args_dict, strict=strict)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {json_path}")
    
    
    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any], strict: bool = False):
        attributes = {**cls.__annotations__, **get_type_hints(cls)}
        keys_to_exclude = list(set(args_dict.keys()) - set(attributes))
        if not strict:
            if keys_to_exclude:
                print(f"Keys {keys_to_exclude} do not match {cls.__name__} attributes and are thus ignored")
            args_dict = {key: value for key, value in args_dict.items() if key in attributes}
        else:
            if keys_to_exclude:
                raise KeyError(f"Unrecognized keys {keys_to_exclude} found in input dictionary")
        keys_defaulted = list(set(attributes) - set(args_dict.keys()))
        if keys_defaulted:
            print(f"Keys {keys_defaulted} were not found in input dictionary and are initialized to default values")
        return cls(**args_dict)
    
    
    def save(self, json_path: Optional[str] = None) -> str:
        if json_path is None:
            json_path = "config.json"
        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=4, default=lambda x: x.__dict__)
        return json_path
