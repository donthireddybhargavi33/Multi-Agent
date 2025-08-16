import re
from typing import Dict, List, Tuple

class ChemistryFormulaParser:
    """Advanced chemistry formula parser with subscript support"""
    
    # Periodic table elements
    ELEMENTS = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
        'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
        # Add more elements as needed
    }
    
    # Common molecules database
    MOLECULES = {
        'H2O': {'name': 'Water', 'type': 'compound'},
        'CH4': {'name': 'Methane', 'type': 'organic'},
        'CH3': {'name': 'Methyl group', 'type': 'radical'},
        'C2H5OH': {'name': 'Ethanol', 'type': 'alcohol'},
        'CO2': {'name': 'Carbon dioxide', 'type': 'compound'},
        'NH3': {'name': 'Ammonia', 'type': 'compound'},
        'H2SO4': {'name': 'Sulfuric acid', 'type': 'acid'},
        'NaCl': {'name': 'Sodium chloride', 'type': 'salt'},
        'C6H12O6': {'name': 'Glucose', 'type': 'sugar'},
        'CaCO3': {'name': 'Calcium carbonate', 'type': 'salt'},
    }
    
    def __init__(self):
        self.formula_pattern = re.compile(r'([A-Z][a-z]?\d*)+')
        self.subscript_pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    
    def parse_formula(self, formula: str) -> Dict:
        """Parse chemical formula and return detailed information"""
        formula = formula.strip()
        
        # Handle subscript notation (CH₃ -> CH3)
        formula = self._normalize_subscripts(formula)
        
        # Check if it's a known molecule
        if formula.upper() in self.MOLECULES:
            return {
                'formula': formula.upper(),
                'name': self.MOLECULES[formula.upper()]['name'],
                'type': self.MOLECULES[formula.upper()]['type'],
                'recognized': True,
                'molecular_weight': self._calculate_molecular_weight(formula.upper())
            }
        
        # Parse as general formula
        elements = self._parse_elements(formula)
        if elements:
            return {
                'formula': formula,
                'elements': elements,
                'molecular_weight': self._calculate_molecular_weight(formula),
                'recognized': len(elements) > 0
            }
        
        return {'formula': formula, 'recognized': False}
    
    def _normalize_subscripts(self, text: str) -> str:
        """Convert subscript notation to regular numbers"""
        subscript_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        
        for sub, num in subscript_map.items():
            text = text.replace(sub, num)
        
        # Handle superscripts for charges
        text = re.sub(r'([+-])(\d*)', r'\1\2', text)
        
        return text
    
    def _parse_elements(self, formula: str) -> Dict[str, int]:
        """Parse formula into element counts"""
        elements = {}
        
        # Simple parser for now - can be enhanced for complex formulas
        matches = self.subscript_pattern.findall(formula)
        
        for element, count in matches:
            if element in self.ELEMENTS:
                count = int(count) if count else 1
                elements[element] = elements.get(element, 0) + count
        
        return elements
    
    def _calculate_molecular_weight(self, formula: str) -> float:
        """Calculate molecular weight from formula"""
        elements = self._parse_elements(formula)
        weight = 0.0
        
        for element, count in elements.items():
            if element in self.ELEMENTS:
                weight += self.ELEMENTS[element] * count
        
        return round(weight, 3)

class PhysicsFormulaParser:
    """Physics formula and constant recognition"""
    
    CONSTANTS = {
        'c': {'value': 299792458, 'unit': 'm/s', 'name': 'Speed of light'},
        'h': {'value': 6.626e-34, 'unit': 'J·s', 'name': 'Planck constant'},
        'g': {'value': 9.81, 'unit': 'm/s²', 'name': 'Gravitational acceleration'},
        'π': {'value': 3.14159, 'unit': '', 'name': 'Pi'},
        'e': {'value': 2.71828, 'unit': '', 'name': 'Euler number'},
        'k': {'value': 1.381e-23, 'unit': 'J/K', 'name': 'Boltzmann constant'},
        'R': {'value': 8.314, 'unit': 'J/(mol·K)', 'name': 'Gas constant'},
    }
    
    FORMULAS = {
        'f=ma': {'name': 'Newton Second Law', 'variables': ['F', 'm', 'a']},
        'e=mc²': {'name': 'Mass-Energy Equivalence', 'variables': ['E', 'm', 'c']},
        'pv=nrt': {'name': 'Ideal Gas Law', 'variables': ['P', 'V', 'n', 'R', 'T']},
        'v=ir': {'name': 'Ohm Law', 'variables': ['V', 'I', 'R']},
        'ke=½mv²': {'name': 'Kinetic Energy', 'variables': ['KE', 'm', 'v']},
        'pe=mgh': {'name': 'Potential Energy', 'variables': ['PE', 'm', 'g', 'h']},
        'p=mv': {'name': 'Momentum', 'variables': ['p', 'm', 'v']},
        'a=v/t': {'name': 'Acceleration', 'variables': ['a', 'v', 't']},
        's=vt': {'name': 'Distance', 'variables': ['s', 'v', 't']},
    }
    
    def __init__(self):
        self.formula_pattern = re.compile(r'([a-zA-Z]+)=([^=]+)')
        self.constant_pattern = re.compile(r'\b([a-z])\b')
    
    def parse_formula(self, formula: str) -> Dict:
        """Parse physics formula and return detailed information"""
        formula = formula.strip().lower()
        
        # Check if it's a known formula
        if formula in self.FORMULAS:
            return {
                'formula': formula,
                'name': self.FORMULAS[formula]['name'],
                'variables': self.FORMULAS[formula]['variables'],
                'recognized': True,
                'type': 'physics_formula'
            }
        
        # Check for constants
        constant_match = self.constant_pattern.match(formula)
        if constant_match and formula in self.CONSTANTS:
            const = self.CONSTANTS[formula]
            return {
                'constant': formula,
                'name': const['name'],
                'value': const['value'],
                'unit': const['unit'],
                'recognized': True,
                'type': 'physics_constant'
            }
        
        return {'formula': formula, 'recognized': False}
    
    def extract_formulas(self, text: str) -> List[Dict]:
        """Extract all physics formulas from text"""
        formulas = []
        
        # Look for formula patterns like E=mc^2
        matches = self.formula_pattern.findall(text)
        for var, expr in matches:
            formula_str = f"{var}={expr}"
            parsed = self.parse_formula(formula_str)
            if parsed['recognized']:
                formulas.append(parsed)
        
        return formulas