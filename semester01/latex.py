class LatexDocument(object):
    begin = r"""
\documentclass[a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage{amsmath}
\begin{document}"""

    end = r'\end{document}'

    def __init__(self):
        self.chunks = []

    def add(self, chunk):
        self.chunks.append(chunk)

    # def _render_equation(self, equation):
        # equation = ' '.join(map(self._render_equation_chunk, equation))
        # return "$$ {} $$".format(equation)

    def _render_chunk(self, chunk):
        if hasattr(chunk, 'to_tex'):
            return chunk.to_tex()
        return str(chunk)

    def _render_chunks(self):
        return '\n'.join(map(self._render_chunk, self.chunks))

    def __str__(self):
        return '\n'.join([
            self.begin, self._render_chunks(), self.end
        ])


class LatexChunk(object):
    before = ''
    after = ''

    def __init__(self, *value):
        self.value = value

    def to_tex(self):
        return ''.join([
            self.before,
            ' '.join(map(self._to_tex, self.value)),
            self.after
        ])

    def _to_tex(self, chunk):
        if hasattr(chunk, 'to_tex'):
            return chunk.to_tex()
        return str(chunk)


class Math(LatexChunk):
    before = '$$'
    after = '$$'


class InlineMath(LatexChunk):
    before = '$'
    after = '$'


class Section(LatexChunk):
    before = '\section*{'
    after = '}'


class Text(LatexChunk):
    before = ''
    after = r'\\'
