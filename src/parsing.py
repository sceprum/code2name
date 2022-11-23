import json
from collections import deque, namedtuple
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import chardet
from omegaconf import DictConfig

Span = namedtuple('Span', ['start', 'stop'])


def extract_methods(path: Path):
    parser = SourceFileParser()
    for match, filepath, signature, body in parser.get_declarations_by_path(path):
        rel_path = filepath.relative_to(path)
        dir_name = rel_path.parts[0]
        filename = filepath.name
        name = _get_method_name(match)
        method = replace_name(signature, match) + body
        entry = {'signature': signature, 'body': body, 'filename': filename, 'dir_name': dir_name, 'name': name,
                 'method': method}
        yield entry


def replace_name(signature, match):
    g = SourceFileParser.name_group_num
    start = match.start(g) - match.start(0)
    end = match.end(g) - match.start(0)
    name = signature[:start] + 'f' + signature[end:]
    return name


def convert_to_jsonl(path: Path, out_file_path):
    with open(out_file_path, 'w') as out_file:
        for row_data in extract_methods(path):
            json_string = json.dumps(row_data)
            out_file.write(json_string)
            out_file.write('\n')


def convert_to_feather(path: Path):
    return pd.DataFrame(extract_methods(path))


def _get_method_name(match):
    return match.group(SourceFileParser.name_group_num)


class SourceFileParser:
    name_group_num = 6

    def __init__(self):
        modifiers_str = 'public|protected|private|static|synchronized|final|abstract|native'
        self._signature_regex = \
            r'((' + modifiers_str +\
            r')\s*)*([\w\.]+)(\<[\w\<\>\[\]?\.,\s]*\>)?(\s*\[\s*\])?\s+(\w+) *\(([\w\<\>\[\]\?.,\s]*)\)[\s\w\.,]+\{'
        #   return type^  arguments of generic^         array^ method name^  method args in brackets^     ^exceptions

        self._signature_pat = re.compile(self._signature_regex)
        self._simple_argument_pat = re.compile(r'[\w\[\]\.\s]+?(\s*\.{3})?\s*([\w\[\]]+)')
        self._type_parameter = re.compile(r'(\<[\w,\s\.\?\[\]]+\>)')
        self._context_blocks = DictConfig({'"': '"', '/*': '*/', '//': '\n'})
        self._block_start_pattern = re.compile(r'(/\*|//|")')
        self._spaces_pattern = re.compile('\s\s+')
        self._modifiers = set(modifiers_str.split('|'))

    def get_valid_modifiers(self):
        return self._modifiers

    def get_declarations_by_path(self, path: Path, extension='*.java'):
        filepaths = list(path.glob('*'))

        if len(filepaths) == 0:
            print(f'No files were found in directory {path}')

        for proj_dir in filepaths:
            if not proj_dir.is_dir():
                continue

            for filepath in tqdm(list(proj_dir.glob(extension)), desc=f'Parsing {proj_dir.name}'):
                # print(filepath)
                yield from self.get_file_declarations(filepath)

    def clear_body(self, s):
        return ' '.join(split_by_pattern(s, self._spaces_pattern))

    def get_file_declarations(self, filepath):
        """
        Iterates method signature and body.
        Regex to find signature is splitted in two to make them simpler and faster to evaluate
        """
        code = read_source_file(filepath)
        for m in self._signature_pat.finditer(code):

            modifier = _get_type(m)
            if self._modifiers and (modifier in self._modifiers):  # modifier instead of type (class constructor)
                continue

            if is_commented_out(code, m):
                continue

            args = _get_method_args(m)
            if not self._matches_method_arguments(args):
                continue

            method_name = _get_method_name(m)
            if (method_name == 'if') or (method_name[0].isupper()):
                continue

            body_start = m.end() - 1
            try:
                body_end = self.find_body_end(code, body_start)
            except RuntimeError:
                continue
            body = code[body_start:body_end + 1]
            body = self.clear_body(body)
            signature = m.group(0)[:-1].strip()
            yield m, filepath, signature, body

    def _matches_method_arguments(self, string):
        string = string.strip()
        if not string:
            return True

        if '<' in string:
            # We don't want to use extra regex in simple case
            if string.count('<') != string.count('>'):
                return False
            parts = split_by_pattern(string, self._type_parameter)

            if len(parts) == 1:
                return False

            simplified = ''.join(parts)
            res = self._matches_method_arguments(simplified)
            return res
        for arg in string.split(','):
            if not self._simple_argument_pat.fullmatch(arg.strip()):
                return False
        return True

    def _find_blocks(self, text: str, start_pos, stop_pos, contexts):
        while start_pos > 0:
            match = self._block_start_pattern.search(text, start_pos, stop_pos)
            if match is None:
                return

            begin = match.start()
            postfix = self._context_blocks.get(match.group(0))
            if match.group(0) == '"':
                if _is_escaped_symbol(text, match.start()):
                    start_pos = match.end()
                    continue
                end = _find_valid_symbol(text, postfix, begin + 1)
            else:
                end = text.find(postfix, begin + 1)
            if end < 0:
                raise RuntimeError('Failed to find end of block')
            else:
                end += len(postfix)
            contexts.append(Span(begin, end))
            start_pos = end

    def find_body_end(self, text, begin_pos):
        if text[begin_pos] != '{':
            raise ValueError('Code at beginning position must contain "{"')
        delta = 1
        close_pos = begin_pos
        open_pos = close_pos

        ignore_blocks = deque()
        while (delta > 0) and (close_pos < len(text)):
            prev_close_pos = close_pos
            close_pos = _find_valid_symbol(text, '}', close_pos + 1)
            if close_pos < 0:
                return -1

            self._find_blocks(text, prev_close_pos, close_pos, ignore_blocks)

            if ignore_blocks and (ignore_blocks[-1].stop > close_pos):
                close_pos = ignore_blocks[-1].stop
                continue

            while True:
                open_pos = _find_valid_symbol(text[:close_pos], '{', open_pos + 1)
                if open_pos < 0:
                    open_pos = close_pos
                    break

                # Pop blocks to the left of the first open bracket
                while ignore_blocks and (ignore_blocks[0].stop < open_pos):
                    ignore_blocks.popleft()

                if not ignore_blocks:
                    delta += 1
                    continue

                block = ignore_blocks[0]
                if open_pos < block.start:  # the right end of the block is laying to the left of the bracket
                    delta += 1

            delta -= 1  # Process current closing bracket

        return close_pos


def is_commented_out(code, match):
    line_start = code[:match.start()].rfind('\n')
    if line_start < 0:
        return False
    return code[line_start + 1:].startswith('//')


def split_by_pattern(string, pat):
    parts = []
    prev = 0
    for m in pat.finditer(string):
        parts.append(string[prev:m.start()])
        prev = m.end()
    parts.append(string[prev:])
    return parts


def _get_method_args(match):
    return match.group(7)


def _get_type(match):
    m = match.group(3)
    return m


def read_source_file(filepath):
    with open(filepath, 'rb') as f:
        file_data = f.read()
        try:
            code = file_data.decode('utf-8')
        except UnicodeDecodeError:
            encoding_info = chardet.detect(file_data)
            code = file_data.decode(encoding_info['encoding'])
    return code


def count_backslashes(text, end_pos):
    if end_pos >= len(text):
        return 0
    count = 0
    for p in range(end_pos, -1, -1):
        if text[p] != '\\':
            return count
        count += 1
    return count


def _is_escaped_symbol(text, symbol_pos):
    prev_char = text[symbol_pos - 1]
    if prev_char not in ('\\', '\''):
        return False
    if prev_char == '\'':
        if not ((len(text) > symbol_pos + 1) and (text[symbol_pos + 1] == '\'')):
            return False
    else:  # prev_char is \\
        if text[symbol_pos - 2: symbol_pos] == '\\\\':
            count = count_backslashes(text, symbol_pos - 1)
            if count % 2 == 0:
                return False
    return True


def _find_valid_symbol(text, symbol, pos):
    """
    Finds symbols  that is not contained in single quotes starting from `pos`
    """
    close_pos = text.find(symbol, pos)
    if close_pos < 0:
        return close_pos
    while (close_pos >= 0) and _is_escaped_symbol(text, close_pos):
        close_pos = text.find(symbol, close_pos + 1)
    return close_pos


def _find_double_quotes(text, pos, positions):
    """
    Adds double quotes found in `text` starting from `pos` to list-like data structure positions
    """
    pos = _find_valid_symbol(text, '"', pos)
    while pos > 0:
        positions.append(pos)
        pos = _find_valid_symbol(text, '"', pos + 1)


def _iterates_blocks_outside_delimiters(code, prefix, postfix):
    start = code.find(prefix)
    while start >= 0:
        part = code[:start]
        yield part
        stop = code.find(postfix)
        if stop < 0:
            break
        code = code[stop + len(postfix):]
        start = code.find(prefix)
    yield code


def print_line_around(text, close_pos):
    stop = start = close_pos
    for i in range(2):
        if start > 0:
            start = text[:start].rfind('\n')
        if stop > 0:
            stop = text.find('\n', stop + 1)

    if (start > 0) and (stop > 0):
        print(text[start + 1: stop])
    else:
        print('Not implemented')


def _print_blocks(text, blocks):
    print([text[b.start:b.stop] for b in blocks])


def parse_raw_data_dir(data_path, splits, save_path):
    save_path = Path(save_path)
    data_path = Path(data_path)
    for split_name in splits:
        split_dir = data_path / split_name
        print(f'Parsing {split_name}...')
        data = convert_to_feather(split_dir)
        if len(data) == 0:
            continue
        split_dir.mkdir(exist_ok=True)
        feather_path = save_path / (split_name + '.feather')
        data.to_feather(feather_path)


def parse_methods(path):
    cols = ('start', 'stop', 'groups', 'filepath', 'signature', 'body')
    methods = deque()

    parser = SourceFileParser()
    for match, filepath, signature, body in parser.get_declarations_by_path(path):
        methods.append((match.start(), match.end(), match.groups(), str(filepath), signature, body))

    methods = pd.DataFrame(methods, columns=cols)
    methods['name'] = methods.groups.apply(lambda x: x[4])
    return methods
