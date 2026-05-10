from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ArgumentSchema:
    required: Dict[str, type] = field(default_factory=dict)
    optional: Dict[str, type] = field(default_factory=dict)

    column_args: List[str] = field(default_factory=list)
    column_list_args: List[str] = field(default_factory=list)
    allow_all_columns: bool = False

    allowed_values: Dict[str, List[Any]] = field(default_factory=dict)
    conditional_allowed_values: Dict[str, Dict[Any, Dict[str, List[Any]]]] = field(
        default_factory=dict
    )
    value_aliases: Dict[str, Dict[Any, Any]] = field(default_factory=dict)

    def canonicalize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(arguments or {})

        for arg_name, alias_map in self.value_aliases.items():
            if arg_name not in args:
                continue

            value = args[arg_name]
            key = value.strip().lower() if isinstance(value, str) else value

            if key in alias_map:
                args[arg_name] = alias_map[key]
            elif isinstance(value, str):
                args[arg_name] = key

        return args

    def to_contract_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "optional": self.optional,
            "column_args": self.column_args,
            "column_list_args": self.column_list_args,
            "allow_all_columns": self.allow_all_columns,
            "allowed_values": self.allowed_values,
            "conditional_allowed_values": self.conditional_allowed_values,
            "value_aliases": self.value_aliases,
        }
