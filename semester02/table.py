from pprint import pprint


class ASCIITable(object):
    def __init__(self, headers, rows=None):
        self.headers = headers
        self.rows = []
        for row in (rows or []):
            self.add_row(row)

    def add_row(self, row):
        assert len(row) == len(self.headers)
        self.rows.append(map(str, row))

    def _horizontal_split(self, width, sym):
        return sym * width + '\n'

    def _format_row(self, row, widths):
        return '|{}|\n'.format('|'.join(
            cell.center(width) for cell, width in zip(row, widths)
        ))

    def __str__(self):
        cols = zip(*([self.headers] + self.rows))
        cols_width = [max(map(len, col)) + 2 for col in cols]
        table_width = sum(cols_width) + len(cols_width) + 1
        result = self._horizontal_split(table_width, '-')
        result += self._format_row(self.headers, cols_width)
        result += self._horizontal_split(table_width, '=')
        for row in self.rows:
            result += self._format_row(row, cols_width)
        result += self._horizontal_split(table_width, '-')
        return result


if __name__ == '__main__':
    table = ASCIITable(['header1', 'header2', 'header3'], [
        ['x11', 'x12', 2],
        ['x21', 2/3., 'x23'],
    ])
    table.add_row(['x31', 'x32', 'x33'])
    print table
