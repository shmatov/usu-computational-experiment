class LatexDocument(object):
    begin = r"""
\documentclass[a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage{amsmath}
\begin{document}"""

    end = r'\end{document}'

    def __init__(self):
        self.equations = []

    def add_equation(self, *equation):
        self.equations.append(equation)

    def _render_equation(self, equation):
        equation = ' '.join(map(self._render_equation_chunk, equation))
        return "$$ {} $$".format(equation)

    def _render_equation_chunk(self, chunk):
        if hasattr(chunk, 'to_tex'):
            return chunk.to_tex()
        return str(chunk)

    def _render_equations(self):
        return '\n'.join(map(self._render_equation, self.equations))

    def __str__(self):
        return '\n'.join([
            self.begin, self._render_equations(), self.end
        ])
