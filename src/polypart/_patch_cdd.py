import fractions

from gmpy2 import mpq

# Force pycddlib to use gmpy2.mpq instead of fractions.Fraction
fractions.Fraction = mpq
