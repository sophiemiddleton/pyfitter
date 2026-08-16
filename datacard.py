"""
Data card system for sensitivity limits (Combine-inspired).
Allows specifying expected yields, shapes, and systematics in a structured format.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DataCard:
    """
    Represents a Combine-like data card for limit setting.
    
    Holds:
    - Expected yields (signal and backgrounds)
    - Systematic uncertainties
    - Kinematic shapes/PDFs
    - Observable ranges
    
    Can be loaded from YAML or JSON format.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.observables = {}  # {'mom': (100, 110), 'time': (0, 1000)}
        self.processes = {}    # {process_name: {'yield': N, 'shape': ...}}
        self.systematics = {}  # {syst_name: {process: effect, ...}}
        self.poi = None        # Parameter of interest (e.g., 'N_CE')
        self.poi_range = None  # (min, max) for POI
        self.metadata = {}     # Extra info
        
    def set_observables(self, obs_dict: Dict[str, Tuple[float, float]]):
        """Set observable ranges: {'mom': (100, 110), 'time': (0, 1000)}"""
        self.observables = obs_dict
        
    def add_process(self, process_name: str, yield_value: float, 
                    shape_type: str = 'free', shape_params: Optional[Dict] = None):
        """
        Add a process (background or signal).
        
        Args:
            process_name: Name of process (e.g., 'DIO', 'Cosmic')
            yield_value: Expected yield (rate)
            shape_type: 'free', 'fixed', 'morphing', or 'template'
            shape_params: Additional shape parameters (depends on shape_type)
        """
        self.processes[process_name] = {
            'yield': yield_value,
            'shape_type': shape_type,
            'shape_params': shape_params or {}
        }
        
    def add_systematic(self, syst_name: str, syst_type: str, 
                      effects: Dict[str, float], syst_params: Optional[Dict] = None):
        """
        Add a systematic uncertainty.
        
        Args:
            syst_name: Name of systematic (e.g., 'scale_uncertainty', 'pdf_shape')
            syst_type: 'lnN' (lognormal), 'shape', 'gmN' (gamma), 'uniform', etc.
            effects: {process_name: effect_value} where effect is e.g., 1.2 for 20% up
            syst_params: Additional parameters
        """
        self.systematics[syst_name] = {
            'type': syst_type,
            'effects': effects,
            'params': syst_params or {}
        }
        
    def set_poi(self, poi_name: str, poi_range: Tuple[float, float]):
        """Set parameter of interest and its range."""
        self.poi = poi_name
        self.poi_range = poi_range
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization. Converts tuples to lists for YAML compatibility."""
        # Convert observables tuples to lists
        obs_dict = {}
        for key, val in self.observables.items():
            obs_dict[key] = list(val) if isinstance(val, tuple) else val
        
        # Convert poi_range tuple to list
        poi_range = list(self.poi_range) if isinstance(self.poi_range, tuple) else self.poi_range
        
        return {
            'name': self.name,
            'observables': obs_dict,
            'processes': self.processes,
            'systematics': self.systematics,
            'poi': self.poi,
            'poi_range': poi_range,
            'metadata': self.metadata
        }
    
    def to_yaml(self, filepath: str):
        """Save data card to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
    def to_json(self, filepath: str):
        """Save data card to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataCard':
        """Load from dictionary. Handles both list and tuple formats."""
        card = cls(data.get('name', 'unnamed'))
        
        # Convert observables lists back to tuples if needed
        obs_data = data.get('observables', {})
        card.observables = {k: tuple(v) if isinstance(v, list) else v 
                           for k, v in obs_data.items()}
        
        card.processes = data.get('processes', {})
        card.systematics = data.get('systematics', {})
        card.poi = data.get('poi')
        
        # Convert poi_range list back to tuple if needed
        poi_range = data.get('poi_range')
        card.poi_range = tuple(poi_range) if isinstance(poi_range, list) else poi_range
        
        card.metadata = data.get('metadata', {})
        return card
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'DataCard':
        """Load from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'DataCard':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_expected_yields(self) -> Dict[str, float]:
        """Extract expected yields for all processes."""
        return {proc: self.processes[proc]['yield'] 
                for proc in self.processes}
    
    def get_process_shape_info(self, process_name: str) -> Dict[str, Any]:
        """Get shape information for a specific process."""
        if process_name not in self.processes:
            return {}
        return {
            'type': self.processes[process_name].get('shape_type', 'free'),
            'params': self.processes[process_name].get('shape_params', {})
        }
    
    def apply_systematic_variation(self, syst_name: str, 
                                   variation: float = 1.0) -> Dict[str, float]:
        """
        Apply systematic variation (e.g., +1sigma).
        Returns modified yields.
        
        Args:
            syst_name: Name of systematic
            variation: Variation level (0=down, 1=nominal, 2=up)
        
        Returns:
            Modified yields dict
        """
        if syst_name not in self.systematics:
            return self.get_expected_yields()
            
        syst = self.systematics[syst_name]
        syst_type = syst['type']
        modified_yields = self.get_expected_yields().copy()
        
        for process, effect in syst['effects'].items():
            if process in modified_yields:
                if syst_type == 'lnN':
                    # Lognormal: effect is 1.2 means ±20%
                    # variation: 0=down (1/1.2), 1=nominal, 2=up (1.2)
                    if variation == 0:
                        modified_yields[process] /= effect
                    elif variation == 2:
                        modified_yields[process] *= effect
                elif syst_type == 'uniform':
                    # Uniform: effect is absolute value
                    modified_yields[process] += effect * (variation - 1)
                    
        return modified_yields
    
    def __repr__(self) -> str:
        return (f"DataCard(name={self.name}, "
                f"processes={list(self.processes.keys())}, "
                f"systematics={list(self.systematics.keys())}, "
                f"poi={self.poi})")


class DataCardCollection:
    """Manages multiple related data cards (e.g., different mass points)."""
    
    def __init__(self, name: str):
        self.name = name
        self.cards = {}  # {card_name: DataCard}
        
    def add_card(self, card: DataCard):
        """Add a data card to collection."""
        self.cards[card.name] = card
        
    def get_card(self, card_name: str) -> Optional[DataCard]:
        """Retrieve a card by name."""
        return self.cards.get(card_name)
    
    def get_all_cards(self) -> Dict[str, DataCard]:
        """Get all cards."""
        return self.cards.copy()


if __name__ == "__main__":
    # Example: Create a simple data card
    card = DataCard("MDS3c_1e-13")
    
    # Define observables
    card.set_observables({
        'mom': (100.0, 110.0),
        'time': (0.0, 1000.0)
    })
    
    # Add processes with expected yields
    card.add_process('DIO', 1429.67, shape_type='free', 
                    shape_params={'model': 'poly58'})
    card.add_process('Cosmic', 329.57, shape_type='free',
                    shape_params={'model': 'chebyshev'})
    card.add_process('RPC', 9.41, shape_type='free',
                    shape_params={'model': 'chebyshev'})
    card.add_process('CE', 0.3375, shape_type='free',
                    shape_params={'model': 'gauss', 'mu': 105.0, 'sigma': 0.5})
    
    # Add systematics
    card.add_systematic('scale_uncertainty', 'lnN',
                       {'DIO': 1.1, 'Cosmic': 1.05, 'RPC': 1.15})
    card.add_systematic('pdf_shape', 'shape',
                       {'DIO': 0.05, 'Cosmic': 0.03})
    
    # Set POI
    card.set_poi('N_CE', (0.0, 40.0))
    
    # Save examples
    card.to_yaml('example_datacard.yaml')
    card.to_json('example_datacard.json')
    
    print(card)
    print("\nExpected yields:", card.get_expected_yields())
    print("\nDIO shape info:", card.get_process_shape_info('DIO'))
