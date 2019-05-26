# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import copy
import re
import sys

from parso import parse
from parso.python.tree import Name, Number, Keyword, Newline, PythonNode
from tri.declarative import evaluate

__version__ = '1.5.0'


class MutationID(object):
    def __init__(self, line, index, line_number):
        self.line = line
        self.index = index
        self.line_number = line_number
        self.mutation_name = "none"

    def __repr__(self):
        return 'MutationID(line="%s", index=%s, line_number=%s, mutation_name=%s)' % (self.line, self.index, self.line_number, self.mutation_name)

    def __eq__(self, other):
        return (self.line, self.index, self.line_number) == (other.line, other.index, other.line_number)


ALL = MutationID(line='%all%', index=-1, line_number=-1)


class InvalidASTPatternException(Exception):
    pass


class ASTPattern(object):
    def __init__(self, source, **definitions):
        if definitions is None:
            definitions = {}
        source = source.strip()

        self.definitions = definitions

        self.module = parse(source)

        self.markers = []

        def get_leaf(line, column, of_type=None):
            r = self.module.children[0].get_leaf_for_position((line, column))
            while of_type is not None and r.type != of_type:
                r = r.parent
            return r

        def parse_markers(node):
            if hasattr(node, '_split_prefix'):
                for x in node._split_prefix():
                    parse_markers(x)

            if hasattr(node, 'children'):
                for x in node.children:
                    parse_markers(x)

            if node.type == 'comment':
                line, column = node.start_pos
                for match in re.finditer(r'\^(?P<value>[^\^]*)', node.value):
                    name = match.groupdict()['value'].strip()
                    d = definitions.get(name, {})
                    assert set(d.keys()) | {'of_type', 'marker_type'} == {'of_type', 'marker_type'}
                    self.markers.append(dict(
                        node=get_leaf(line - 1, column + match.start(), of_type=d.get('of_type')),
                        marker_type=d.get('marker_type'),
                        name=name,
                    ))

        parse_markers(self.module)

        pattern_nodes = [x['node'] for x in self.markers if x['name'] == 'match' or x['name'] == '']
        if len(pattern_nodes) != 1:
            raise InvalidASTPatternException("Found more than one match node. Match nodes are nodes with an empty name or with the explicit name 'match'")
        self.pattern = pattern_nodes[0]
        self.marker_type_by_id = {id(x['node']): x['marker_type'] for x in self.markers}

    def matches(self, node, pattern=None, skip_child=None):
        if pattern is None:
            pattern = self.pattern

        check_value = True
        check_children = True

        # Match type based on the name, so _keyword matches all keywords. Special case for _all that matches everything
        if pattern.type == 'name' and pattern.value.startswith('_') and pattern.value[1:] in ('any', node.type):
            check_value = False

        # The advanced case where we've explicitly marked up a node with the accepted types
        elif id(pattern) in self.marker_type_by_id:
            if self.marker_type_by_id[id(pattern)] in (pattern.type, 'any'):
                check_value = False
                check_children = False  # TODO: really? or just do this for 'any'?

        # Check node type strictly
        elif pattern.type != node.type:
            return False

        # Match children
        if check_children and hasattr(pattern, 'children'):
            if len(pattern.children) != len(node.children):
                return False

            for pattern_child, node_child in zip(pattern.children, node.children):
                if node_child is skip_child:  # prevent infinite recursion
                    continue

                if not self.matches(node=node_child, pattern=pattern_child, skip_child=node_child):
                    return False

        # Node value
        if check_value and hasattr(pattern, 'value'):
            if pattern.value != node.value:
                return False

        # Parent
        if pattern.parent.type != 'file_input':  # top level matches nothing
            if skip_child != node:
                return self.matches(node=node.parent, pattern=pattern.parent, skip_child=node)

        return True


def full_statement_from_node(node):
    while node.parent and not node.type.endswith('_stmt') and not node.type.endswith('def') and node.type not in ('suite', 'endmarker'):
        node = node.parent

    return node.get_code()


# We have a global whitelist for constants of the pattern __all__, __version__, etc

dunder_whitelist = [
    'all',
    'version',
    'title',
    'package_name',
    'author',
    'description',
    'email',
    'version',
    'license',
    'copyright',
]


if sys.version_info < (3, 0):   # pragma: no cover (python 2 specific)
    # noinspection PyUnresolvedReferences
    text_types = (str, unicode)  # noqa: F821
else:
    text_types = (str,)


UNTESTED = 'untested'
OK_KILLED = 'ok_killed'
OK_SUSPICIOUS = 'ok_suspicious'
BAD_TIMEOUT = 'bad_timeout'
BAD_SURVIVED = 'bad_survived'


mutant_statuses = [
    UNTESTED,
    OK_KILLED,
    OK_SUSPICIOUS,
    BAD_TIMEOUT,
    BAD_SURVIVED,
]


def number_mutation(value, **_):
    suffix = ''
    if value.upper().endswith('L'):  # pragma: no cover (python 2 specific)
        suffix = value[-1]
        value = value[:-1]

    if value.upper().endswith('J'):
        suffix = value[-1]
        value = value[:-1]

    if value.startswith('0o'):
        base = 8
        value = value[2:]
    elif value.startswith('0x'):
        base = 16
        value = value[2:]
    elif value.startswith('0b'):
        base = 2
        value = value[2:]
    elif value.startswith('0') and len(value) > 1 and value[1] != '.':  # pragma: no cover (python 2 specific)
        base = 8
        value = value[1:]
    else:
        base = 10

    try:
        parsed = int(value, base=base)
    except ValueError:
        # Since it wasn't an int, it must be a float
        parsed = float(value)

    result = repr(parsed + 1)
    if not result.endswith(suffix):
        result += suffix
    return result


def string_mutation(value, **_):
    prefix = value[:min([x for x in [value.find('"'), value.find("'")] if x != -1])]
    value = value[len(prefix):]

    if value.startswith('"""') or value.startswith("'''"):
        # We assume here that triple-quoted stuff are docs or other things
        # that mutation is meaningless for
        return prefix + value
    return prefix + value[0] + 'XX' + value[1:-1] + 'XX' + value[-1]


def partition_node_list(nodes, value):
    for i, n in enumerate(nodes):
        if hasattr(n, 'value') and n.value == value:
            return nodes[:i], n, nodes[i + 1:]

    assert False, "didn't find node to split on"


def lambda_mutation(children, **_):
    pre, op, post = partition_node_list(children, value=':')

    if len(post) == 1 and getattr(post[0], 'value', None) == 'None':
        return pre + [op] + [Number(value=' 0', start_pos=post[0].start_pos)]
    else:
        return pre + [op] + [Keyword(value=' None', start_pos=post[0].start_pos)]


NEWLINE = {'formatting': [], 'indent': '', 'type': 'endl', 'value': ''}


def argument_mutation(children, context, **_):
    """
    :type context: Context
    """
    if len(context.stack) >= 3 and context.stack[-3].type in ('power', 'atom_expr'):
        stack_pos_of_power_node = -3
    elif len(context.stack) >= 4 and context.stack[-4].type in ('power', 'atom_expr'):
        stack_pos_of_power_node = -4
    else:
        return

    power_node = context.stack[stack_pos_of_power_node]

    if power_node.children[0].type == 'name' and power_node.children[0].value in context.dict_synonyms:
        c = children[0]
        if c.type == 'name':
            children = children[:]
            children[0] = Name(c.value + 'XX', start_pos=c.start_pos, prefix=c.prefix)
            return children

# Mutates 'raise' or 'raise <exception>' to 'pass'
def raise_mutation(children, node, **_):
    children = children[:]
    if len(children) <= 2:
        if len(children) == 2:
            children.pop()
        c = children[0]
        children[0] = Keyword(value='pass', start_pos=node.start_pos, prefix=c.prefix)
    return children



def keyword_mutation(value, context, **_):

    if len(context.stack) > 2 and context.stack[-2].type == 'comp_op' and value in ('in', 'is'):
        return

    if len(context.stack) > 1 and context.stack[-2].type == 'for_stmt':
        return

    return {
        # 'not': 'not not',
        'not': '',
        'is': 'is not',  # this will cause "is not not" sometimes, so there's a hack to fix that later
        'in': 'not in',
        'break': 'continue',
        'continue': 'break',
        'True': 'False',
        'False': 'True',
    }.get(value)


import_from_star_pattern = ASTPattern("""
from _name import *
#                 ^
""")

def operator_mutation(value, node, **_):
    if import_from_star_pattern.matches(node=node):
        return

    if value in ('*', '**') and node.parent.type == 'param':
        return

    if value == '*' and node.parent.type == 'parameters':
        return

    if value in ('*', '**') and node.parent.type in ('argument', 'arglist'):
        return

    return {
        '+': '-',
        '-': '+',
        '*': {'div':'/', 'exp':'**'},
        '/': '*',
        '//': '/',
        '%': '/',
        '<<': '>>',
        '>>': '<<',
        '&': '|',
        '|': '&',
        '^': '&',
        '**': '*',
        '~': '',

        '+=': '-=',
        '-=': '+=',
        '*=': '/=',
        '/=': '*=',
        '//=': '/=',
        '%=': '/=',
        '<<=': '>>=',
        '>>=': '<<=',
        '&=': '|=',
        '|=': '&=',
        '^=': '&=',
        '**=': '*=',
        '~=': '=',

        '<': '<=',
        '<=': '<',
        '>': '>=',
        '>=': '>',
        '==': '!=',
        '!=': '==',
        '<>': '==',
    }.get(value)


def and_or_test_mutation(children, node, **_):
    children = children[:]
    children[1] = Keyword(
        value={'and': ' or', 'or': ' and'}[children[1].value],
        start_pos=node.start_pos,
    )
    return children


def expression_mutation(children, **_):
    def handle_assignment(children):
        if getattr(children[2], 'value', '---') != 'None':
            x = ' None'
        else:
            x = ' ""'
        children = children[:]
        children[2] = Name(value=x, start_pos=children[2].start_pos)

        return children

    if children[0].type == 'operator' and children[0].value == ':':
        if len(children) > 2 and children[2].value == '=':
            children[1:] = handle_assignment(children[1:])
    elif children[1].type == 'operator' and children[1].value == '=':
        children = handle_assignment(children)

    return children


def decorator_mutation(children, **_):
    assert children[-1].type == 'newline'
    return children[-1:]


array_subscript_pattern = ASTPattern("""
_name[_any]
#       ^
""")


function_call_pattern = ASTPattern("""
_name(_any)
#       ^
""")


def name_mutation(node, value, **_):
    simple_mutants = {
        'True': 'False',
        'False': 'True',
        'deepcopy': 'copy',
        'None': '""',
        # TODO: probably need to add a lot of things here... some builtins maybe, what more?
    }
    if value in simple_mutants:
        return simple_mutants[value]

    if array_subscript_pattern.matches(node=node):
        return 'None'

    if function_call_pattern.matches(node=node):
        return 'None'

def loop_mutation(children, node, **_):
    """
    Mutates loop

    For loop children is structured as the nodes that make up the loop defintion (for x in y:) and a suite
    The suite is a newline, indented, statement, then dedent and another newline
    """
    # for x in y
    # node.get_defined_names() = x
    # node.get_testlist() = y

    mutations = {}
    # need to deep copy inside subfunctions, otherwise both mutations applied at same time
    # but the deep copy breaks that new != old check in mutate_node, will throw the source assert in mutate()
    if node.type == 'for_stmt':
        mutations['zero'] = zero_loop_mutation_for(children, node, **_)
    elif node.type == 'while_stmt':
        mutations['zero'] = zero_loop_mutation_while(children, node, **_)
    mutations['one'] = one_loop_mutation(children, node, **_)

    return mutations


def zero_loop_mutation_for(base_children, node, **_):
    """
    Zero-run loop: Replaces iteration list with a blank list []
        Surviving Test Case: Only write test case that checks for an empty loop, eg input loop = []
        Could also do this by appending a break as first element
    """
    children = copy.deepcopy(base_children)
    empty_loop = ' []'

    testlist = node.get_testlist()

    for idx, c in enumerate(children):
        if c.start_pos == testlist.start_pos: # deepcopy messes the comparision, use position instead
            children[idx] = Name(
                    value=empty_loop,
                    start_pos=testlist.start_pos)
            break

    return children

def zero_loop_mutation_while(base_children, node, **_):
    """
    Zero-run loop: Adds break statement as first item in loop body
        Surviving Test Case: Only write test case that checks for an empty loop, eg input loop = []
        TODO: Consider merging the for loop mutation with this one
    """
    children = copy.deepcopy(base_children)
    suite = get_loop_body(children)

    # assume 0th element of loop body is a newline
    # assume 1st element of loop body is actual start of function, with proper indentation
    # it seems that line positions are pointless?!
    break_node = create_break_node(suite[1].start_pos)
    suite.insert(1, break_node)

    return children

def one_loop_mutation(base_children, node, **_):
    """
    One-run loop: Adds break statement at the end, so that always gets hit on first run (excluding early return)
        keyword mutation handles replacing continues with break
        Surviving Test Case: test with intended single run loop, eg for x in [0]
    """
    children = copy.deepcopy(base_children)

    for idx, c in enumerate(children):
        # suite is the actual body of the loop
        if c.type == 'suite':
            # add a break
            suite = c
            suite_children = c.children[1:] # assume first element in suite is a newline
            indent_col = suite_children[0].start_pos[1]

            last_element = c.get_last_leaf()
            if last_element.type != 'newline':
                # I'm assuming the last element will always be newline.
                # If it isn't, how on earth is the next line written?
                # Maybe an edge case at the very end of the file?
                return None
            # new break position is at next line, but keep the column indent
            line = last_element.end_pos[0]
            new_pos = (line, indent_col)

            break_node = create_break_node(new_pos)

            suite.children.append(break_node)
            break
    return children

def get_loop_body(children):
    # this may also be children[-1]
    for idx, c in enumerate(children):
        if c.type == 'suite':
            return c.children
    return None

def create_break_node(pos):
    """ Creates a break node.
        pos: (line, column)
    """
    # prefix needs to be added to the first element
    prefix = ' ' * pos[1]
    kw_node = Keyword(value='break', start_pos=pos, prefix=prefix)
    nl_node = Newline(value='\n', start_pos=kw_node.end_pos)
    break_node = PythonNode('simple_stmt', [kw_node, nl_node])
    return break_node

def list_comprehension_mutation(children, node, **_):
    # for exprlist in or_test [comp_iter]
    # find in, everything asfter that is the list
    children = children[:]
    list_idx = None
    for idx, child in enumerate(children):
        if child.type == 'keyword' and child.value == 'in':
            list_idx = idx + 1
            break
    if not list_idx:
        return None
    # assuming the list is of type Name
    empty_list = Name(value=' []', start_pos=children[list_idx].start_pos)
    children[list_idx] = empty_list
    return children[:list_idx+1]

def create_pass_simple_stmt(pos):
    return create_keyword_simple_stmt('pass', pos)

def create_raise_simple_stmt(pos):
    return create_keyword_simple_stmt('raise', pos)

def create_keyword_simple_stmt(keyword, pos):
    """ Creates a simple statement containing one keyword.
        pos: (line, column) at which to start
    """
    kw_node = Keyword(value=keyword, start_pos=pos, prefix=' ')
    nl_node = Newline(value='\n', start_pos=kw_node.end_pos)
    pass_simple_stmt_node = PythonNode('simple_stmt', [kw_node, nl_node])
    return pass_simple_stmt_node

def locate_next_suite_or_stmt_index(children, start_index):
    i = start_index
    while i < len(children):
        node = children[i]
        if type(node) == PythonNode and node.type in ['suite', 'simple_stmt']:
            return i
        i += 1
    return -1

def try_block_mutation_helper(children, marker_clause_index, stmt_creator_fn):
    """Helper method to reduce code duplication in the try_stmt mutator."""

    # Find the index of the statement or suite which is the block controlled by the try block sub-clause
    suite_index = locate_next_suite_or_stmt_index(children, marker_clause_index)
    if suite_index >= 0:
        # Replace the block with a mutation; either a pass or a raise.
        new_children = copy.deepcopy(children)
        pos = new_children[suite_index].start_pos
        mutated_node = stmt_creator_fn(pos)
        new_children[suite_index] = mutated_node
        return new_children

def try_stmt_mutation(children, **_):
    # Locate child indexes for the except, else, and finally blocks
    # Note: else_ and finally_ index should only ever hold one element but
    # are lists for coding convenience
    except_indexes = [i for i, x in enumerate(children) if type(x) == PythonNode and x.type == 'except_clause']

    index_else = [i for i, x in enumerate(children) if type(x) == Keyword and x.value == 'else']
    assert len(index_else) <= 1

    index_finally = [i for i, x in enumerate(children) if type(x) == Keyword and x.value == 'finally']
    assert len(index_finally) <= 1

    mutations = {}
    for i in except_indexes:
        # except => pass
        new_children = try_block_mutation_helper(children, i, create_pass_simple_stmt)
        if new_children:
            key = "except-pass-" + str(i)
            mutations[key] = new_children

        # except => raise
        new_children = try_block_mutation_helper(children, i, create_raise_simple_stmt)
        if new_children:
            key = "except-raise-" + str(i)
            mutations[key] = new_children

    for i in index_else:
        # else => pass
        new_children = try_block_mutation_helper(children, i, create_pass_simple_stmt)
        if new_children:
            mutations["else"] = new_children
        break

    for i in index_finally:
        # finally => pass
        new_children = try_block_mutation_helper(children, i, create_pass_simple_stmt)
        if new_children:
            mutations["finally"] = new_children
        break

    return mutations

mutations_by_type = {
    'operator': dict(value=operator_mutation),
    'keyword': dict(value=keyword_mutation),
    'number': dict(value=number_mutation),
    'name': dict(value=name_mutation),
    'string': dict(value=string_mutation),
    'argument': dict(children=argument_mutation),
    'or_test': dict(children=and_or_test_mutation),
    'and_test': dict(children=and_or_test_mutation),
    'lambdef': dict(children=lambda_mutation),
    'expr_stmt': dict(children=expression_mutation),
    'decorator': dict(children=decorator_mutation),
    'annassign': dict(children=expression_mutation),
    'for_stmt': dict(children=loop_mutation),
    'while_stmt': dict(children=loop_mutation),
    'comp_for': dict(children=list_comprehension_mutation),
    'raise_stmt': dict(children=raise_mutation),
    'try_stmt': dict(children=try_stmt_mutation),
}

# TODO: detect regexes and mutate them in nasty ways? Maybe mutate all strings as if they are regexes


class Context(object):
    def __init__(self, source=None, mutation_id=ALL, dict_synonyms=None, filename=None, exclude=lambda context: False, config=None):
        self.index = 0
        self.remove_newline_at_end = False
        if source and source[-1] != '\n':
            source += '\n'
            self.remove_newline_at_end = True
        self.source = source
        self.mutation_id = mutation_id
        self.number_of_performed_mutations = 0
        self.performed_mutation_ids = []
        assert isinstance(mutation_id, MutationID)
        self.current_line_index = 0
        self.filename = filename
        self.exclude = exclude
        self.stack = []
        self.dict_synonyms = (dict_synonyms or []) + ['dict']
        self._source_by_line_number = None
        self._pragma_no_mutate_lines = None
        self._path_by_line = None
        self.config = config

    def exclude_line(self):
        current_statement = full_statement_from_node(self.stack[-1]).strip()

        if current_statement.startswith('__'):
            word, _, rest = current_statement[2:].partition('__')
            if word in dunder_whitelist and rest.strip()[0] == '=':
                return True

        if current_statement == "__import__('pkg_resources').declare_namespace(__name__)":
            return True

        return self.current_line_index in self.pragma_no_mutate_lines or self.exclude(context=self)

    @property
    def source_by_line_number(self):
        if self._source_by_line_number is None:
            self._source_by_line_number = self.source.split('\n')
        return self._source_by_line_number

    @property
    def current_source_line(self):
        return self.source_by_line_number[self.current_line_index]

    @property
    def mutation_id_of_current_index(self):
        return MutationID(line=self.current_source_line, index=self.index, line_number=self.current_line_index)

    @property
    def pragma_no_mutate_lines(self):
        if self._pragma_no_mutate_lines is None:
            self._pragma_no_mutate_lines = {
                i
                for i, line in enumerate(self.source_by_line_number)
                if '# pragma:' in line and 'no mutate' in line.partition('# pragma:')[-1]
            }
        return self._pragma_no_mutate_lines

    def should_mutate(self):
        if self.mutation_id == ALL:
            return True
        return self.mutation_id in (ALL, self.mutation_id_of_current_index)


def mutate(context):
    """
    :type context: Context
    :return: tuple: mutated source code, number of mutations performed
    :rtype: tuple[str, int]
    """
    try:
        result = parse(context.source, error_recovery=False)
    except Exception:
        print('Failed to parse %s. Internal error from parso follows.' % context.filename)
        print('----------------------------------')
        raise
    mutate_list_of_nodes(result, context=context)
    mutated_source = result.get_code().replace(' not not ', ' ')
    if context.remove_newline_at_end:
        assert mutated_source[-1] == '\n'
        mutated_source = mutated_source[:-1]
    if context.number_of_performed_mutations:
        # If we said we mutated the code, check that it has actually changed
        assert context.source != mutated_source
    context.mutated_source = mutated_source
    return mutated_source, context.number_of_performed_mutations


def mutate_node(node, context):
    """
    :type context: Context
    """
    context.stack.append(node)
    try:
        if node.type in ('tfpdef', 'import_from', 'import_name'):
            return

        if node.start_pos[0] - 1 != context.current_line_index:
            context.current_line_index = node.start_pos[0] - 1
            context.index = 0  # indexes are unique per line, so start over here!

        if hasattr(node, 'children'):
            mutate_list_of_nodes(node, context=context)

            # this is just an optimization to stop early
            if context.number_of_performed_mutations and context.mutation_id != ALL:
                return

        mutation = mutations_by_type.get(node.type)

        if mutation is None:
            return

        for key, value in sorted(mutation.items()):
            old = getattr(node, key)
            if context.exclude_line():
                continue

            new_evaluation = evaluate(
                value,
                context=context,
                node=node,
                value=getattr(node, 'value', None),
                children=getattr(node, 'children', None),
            )

            # This is a dict because lists are used for children
            # I guess a set might be fine too, best would be a custom data struct
            new_mutations = []
            if type(new_evaluation) == dict:
                new_mutations = new_evaluation.values()
            else:
                new_mutations.append(new_evaluation)

            for new in new_mutations:
                assert not callable(new)
                if new is not None and new != old:
                    if context.should_mutate():
                        context.number_of_performed_mutations += 1
                        mutation_id = context.mutation_id_of_current_index
                        mutation_id.mutation_name = node.type
                        context.performed_mutation_ids.append(mutation_id)
                        setattr(node, key, new)
                    context.index += 1

                # this is just an optimization to stop early
                if context.number_of_performed_mutations and context.mutation_id != ALL:
                    return

    finally:
        context.stack.pop()


def mutate_list_of_nodes(node, context):
    """
    :type context: Context
    """

    return_annotation_started = False

    for child_node in node.children:

        if child_node.type == 'operator' and child_node.value == '->':
            return_annotation_started = True

        if return_annotation_started and child_node.type == 'operator' and child_node.value == ':':
            return_annotation_started = False

        if return_annotation_started:
            continue

        mutate_node(child_node, context=context)

        # this is just an optimization to stop early
        if context.number_of_performed_mutations and context.mutation_id != ALL:
            return


def count_mutations(context):
    """
    :type context: Context
    """
    assert context.mutation_id == ALL
    mutate(context)
    return context.number_of_performed_mutations


def list_mutations(context):
    """
    :type context: Context
    """
    assert context.mutation_id == ALL
    mutate(context)
    return context.performed_mutation_ids


def mutate_file(backup, context):
    """
    :type backup: bool
    :type context: Context
    """
    with open(context.filename) as f:
        code = f.read()
    context.source = code
    if backup:
        with open(context.filename + '.bak', 'w') as f:
            f.write(code)
    result, number_of_mutations_performed = mutate(context)
    with open(context.filename, 'w') as f:
        f.write(result)
    return number_of_mutations_performed
