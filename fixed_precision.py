from decimal import Decimal, ROUND_DOWN


class Num(Decimal):
    precision = 10

    def __new__(cls, value):
        if isinstance(value, float):
            value = cls._quantize(Decimal(value))

        return super(Num, cls).__new__(cls, value)

    @classmethod
    def _quantize(self, value):
        if isinstance(value, Decimal):
            decimal_precision = Decimal('.' + '0' * self.precision)
            return Num(value.quantize(decimal_precision, rounding=ROUND_DOWN))
        return value

    def __mul__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__mul__(*args, **kwargs))

    def __add__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__add__(*args, **kwargs))

    def __sub__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__sub__(*args, **kwargs))

    def __div__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__div__(*args, **kwargs))

    def __mod__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__mod__(*args, **kwargs))

    def __pow__(self, *args, **kwargs):
        return self._quantize(super(Num, self).__pow__(*args, **kwargs))

    def __repr__(self):
        return 'Num({})'.format(self)
