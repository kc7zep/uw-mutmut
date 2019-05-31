# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys

from parso import parse

from mutmut import mutate, count_mutations, ALL, Context, list_mutations, MutationID, array_subscript_pattern, function_call_pattern, ASTPattern
import pytest


@pytest.mark.skipif(sys.version_info < (3, 0), reason="Don't check Python 3 syntax in Python 2")
def test_matches_py3():
    node = parse('a: Optional[int] = 7\n').children[0].children[0].children[1].children[1].children[1].children[1]
    assert not array_subscript_pattern.matches(node=node)


def test_matches():
    node = parse('from foo import bar').children[0]
    assert not array_subscript_pattern.matches(node=node)
    assert not function_call_pattern.matches(node=node)
    assert not array_subscript_pattern.matches(node=node)
    assert not function_call_pattern.matches(node=node)

    node = parse('foo[bar]\n').children[0].children[0].children[1].children[1]
    assert array_subscript_pattern.matches(node=node)

    node = parse('foo(bar)\n').children[0].children[0].children[1].children[1]
    assert function_call_pattern.matches(node=node)


def test_ast_pattern_for_loop():
    p = ASTPattern(
        """
for x in y:
#   ^ n  ^ match
    pass
    # ^ x
""",
        x=dict(
            of_type='simple_stmt',
            marker_type='any',
        ),
        n=dict(
            marker_type='name',
        ),
        match=dict(
            marker_type='any',
        )
    )

    n = parse("""for a in [1, 2, 3]:
    if foo:
        continue
""").children[0].children[3]
    assert p.matches(node=n)

    n = parse("""for a, b in [1, 2, 3]:
    if foo:
        continue
""").children[0].children[3]
    assert p.matches(node=n)


@pytest.mark.parametrize(
    'original, expected', [
        ('lambda: 0', 'lambda: None'),
        ('a(b)', 'a(None)'),
        ('a[b]', 'a[None]'),
        ("1 in (1, 2)", "2 not in (2, 3)"),
        ('1+1', '2-2'),
        ('1', '2'),
        ('1-1', '2+2'),
        ('1/1', '2*2'),
        # ('1.0', '1.0000000000000002'),  # using numpy features
        ('1.0', '2.0'),
        ('0.1', '1.1'),
        ('1e-3', '1.001'),
        ('True', 'False'),
        ('False', 'True'),
        ('"foo"', '"XXfooXX"'),
        ("'foo'", "'XXfooXX'"),
        ("u'foo'", "u'XXfooXX'"),
        ("0", "1"),
        ("0o0", "1"),
        ("0.", "1.0"),
        ("0x0", "1"),
        ("0b0", "1"),
        ("1<2", "2<=3"),
        ('(1, 2)', '(2, 3)'),
        ("1 not in (1, 2)", "2  in (2, 3)"),  # two spaces here because "not in" is two words
        ("foo is foo", "foo is not foo"),
        ("foo is not foo", "foo is  foo"),
        ("x if a else b", "x if a else b"),
        ('a or b', 'a and b'),
        ('a and b', 'a or b'),
        ('a = b', 'a = None'),
        ('s[0]', 's[1]'),
        ('s[0] = a', 's[1] = None'),
        ('s[x]', 's[None]'),
        ('s[1:]', ['s[2:]', 's[:]', 's[1:-1]']),
        ('1j', '2j'),
        ('1.0j', '2.0j'),
        ('0o1', '2'),
        ('1.0e10', '10000000001.0'),
        ("dict(a=b)", "dict(aXX=b)"),
        ("Struct(a=b)", "Struct(aXX=b)"),
        ("FooBarDict(a=b)", "FooBarDict(aXX=b)"),
        ('lambda **kwargs: Variable.integer(**setdefaults(kwargs, dict(show=False)))', 'lambda **kwargs: None'),
        ('a = {x for x in y}', 'a = None'),
        ('break', 'continue'),
        ('raise Exception("foo")', 'pass')
    ]
)
def test_basic_mutations(original, expected):
    actual, number_of_performed_mutations = mutate(Context(source=original, mutation_id=ALL, dict_synonyms=['Struct', 'FooBarDict']))
    if type(expected) == list:
        assert actual in expected
        assert number_of_performed_mutations == len(expected), 'Performed %s mutations for original "%s"' % (number_of_performed_mutations, original)
    else:
        assert actual == expected, 'Performed %s mutations for original "%s"' % (number_of_performed_mutations, original)

@pytest.mark.parametrize(
    'original, expected', [
        ('1*1', ['1/1', '1**1']),
        ('for x in y:\n    foo(x)\n', [
            'for x in []:\n    foo(x)\n',
            'for x in y:\n    foo(x)\n    break\n',
            ]),
        ('while x:\n    foo(x)\n', [
            'while x:\n    break\n    foo(x)\n',
            'while x:\n    foo(x)\n    break\n',
            ]),
    ]
)

def test_basic_multiple_mutations(original, expected):
    """
    This test is for multiple mutations per operator. 
    These mutations will clobber each other in the ALL mutations test.
    Because other types of mutation might also happen, we check for each expected mutation being in the list of actual mutations.
    """
    actual = list_mutations(Context(source=original))
    actual_mutation_code = []
    for m in actual:
        mutation, num_mutations = mutate(Context(source=original, mutation_id=m))
        assert num_mutations == 1
        actual_mutation_code.append(mutation)
    for m in expected:
        assert m in actual_mutation_code


@pytest.mark.skipif(sys.version_info < (3, 0), reason="Don't check Python 3 syntax in Python 2")
@pytest.mark.parametrize(
    'original, expected', [
        ('a: int = 1', 'a: int = None'),
        ('a: Optional[int] = None', 'a: Optional[int] = ""'),
        ('def foo(s: Int = 1): pass', 'def foo(s: Int = 2): pass'),
        ('a = None', 'a = ""'),
        ('lambda **kwargs: None', 'lambda **kwargs: 0'),
        ('lambda: None', 'lambda: 0'),
    ]
)
def test_basic_mutations_python3(original, expected):
    actual = mutate(Context(source=original, mutation_id=ALL, dict_synonyms=['Struct', 'FooBarDict']))[0]
    assert actual == expected


@pytest.mark.skipif(sys.version_info < (3, 6), reason="Don't check Python 3.6+ syntax in Python < 3.6")
@pytest.mark.parametrize(
    'original, expected', [
        ('a: int = 1', 'a: int = None'),
        ('a: Optional[int] = None', 'a: Optional[int] = ""'),
    ]
)
def test_basic_mutations_python36(original, expected):
    actual = mutate(Context(source=original, mutation_id=ALL, dict_synonyms=['Struct', 'FooBarDict']))[0]
    assert actual == expected


@pytest.mark.parametrize(
    'source', [
        'foo(a, *args, **kwargs)',
        "'''foo'''",  # don't mutate things we assume to be docstrings
        "r'''foo'''",  # don't mutate things we assume to be docstrings
        "NotADictSynonym(a=b)",
        'from foo import *',
        'from .foo import *',
        'import foo',
        'import foo as bar',
        'foo.bar',
        'for x in y: pass',
        'def foo(a, *args, **kwargs): pass',
        'import foo',
    ]
)
def test_do_not_mutate(source):
    actual = mutate(Context(source=source, mutation_id=ALL, dict_synonyms=['Struct', 'FooBarDict']))[0]
    assert actual == source


@pytest.mark.skipif(sys.version_info < (3, 0), reason="Don't check Python 3 syntax in Python 2")
@pytest.mark.parametrize(
    'source', [
        'def foo(s: str): pass',
        'def foo(a, *, b): pass',
        'a[None]',
        'a(None)',
    ]
)
def test_do_not_mutate_python3(source):
    actual = mutate(Context(source=source, mutation_id=ALL, dict_synonyms=['Struct', 'FooBarDict']))[0]
    assert actual == source


@pytest.mark.skipif(sys.version_info < (3, 0), reason="Don't check Python 3 syntax in Python 2")
def test_mutate_body_of_function_with_return_type_annotation():
    source = """
def foo() -> int:
    return 0
    """

    assert mutate(Context(source=source, mutation_id=ALL))[0] == source.replace('0', '1')


def test_mutate_all():
    assert mutate(Context(source='def foo():\n    return 1+1', mutation_id=ALL)) == ('def foo():\n    return 2-2', 3)


def test_mutate_both():
    source = 'a = b + c'
    mutations = list_mutations(Context(source=source))
    assert len(mutations) == 2
    assert mutate(Context(source=source, mutation_id=mutations[0])) == ('a = b - c', 1)
    assert mutate(Context(source=source, mutation_id=mutations[1])) == ('a = None', 1)


def test_count_available_mutations():
    assert count_mutations(Context(source='def foo():\n    return 1+1')) == 3


def test_perform_one_indexed_mutation():
    assert mutate(Context(source='1+1', mutation_id=MutationID(line='1+1', index=0, line_number=0))) == ('2+1', 1)
    assert mutate(Context(source='1+1', mutation_id=MutationID('1+1', 1, line_number=0))) == ('1-1', 1)
    assert mutate(Context(source='1+1', mutation_id=MutationID('1+1', 2, line_number=0))) == ('1+2', 1)

    # TODO: should this case raise an exception?
    # assert mutate(Context(source='def foo():\n    return 1', mutation_id=2)) == ('def foo():\n    return 1\n', 0)


def test_function():
    source = "def capitalize(s):\n    return s[0].upper() + s[1:] if s else s\n"
    assert mutate(Context(source=source, mutation_id=MutationID(source.split('\n')[1], 0, line_number=1))) == ("def capitalize(s):\n    return s[1].upper() + s[1:] if s else s\n", 1)
    assert mutate(Context(source=source, mutation_id=MutationID(source.split('\n')[1], 1, line_number=1))) == ("def capitalize(s):\n    return s[0].upper() - s[1:] if s else s\n", 1)
    assert mutate(Context(source=source, mutation_id=MutationID(source.split('\n')[1], 2, line_number=1))) == ("def capitalize(s):\n    return s[0].upper() + s[2:] if s else s\n", 1)


@pytest.mark.skipif(sys.version_info < (3, 0), reason="Don't check Python 3 syntax in Python 2")
def test_function_with_annotation():
    source = "def capitalize(s : str):\n    return s[0].upper() + s[1:] if s else s\n"
    assert mutate(Context(source=source, mutation_id=MutationID(source.split('\n')[1], 0, line_number=1))) == ("def capitalize(s : str):\n    return s[1].upper() + s[1:] if s else s\n", 1)


def test_pragma_no_mutate():
    source = """def foo():\n    return 1+1  # pragma: no mutate\n"""
    assert mutate(Context(source=source, mutation_id=ALL)) == (source, 0)


def test_pragma_no_mutate_and_no_cover():
    source = """def foo():\n    return 1+1  # pragma: no cover, no mutate\n"""
    assert mutate(Context(source=source, mutation_id=ALL)) == (source, 0)


def test_mutate_decorator():
    source = """@foo\ndef foo():\n    pass\n"""
    assert mutate(Context(source=source, mutation_id=ALL)) == (source.replace('@foo', ''), 1)


# TODO: getting this test and the above to both pass is tricky
# def test_mutate_decorator2():
#     source = """\"""foo\"""\n\n@foo\ndef foo():\n    pass\n"""
#     assert mutate(Context(source=source, mutation_id=ALL)) == (source.replace('@foo', ''), 1)


def test_mutate_dict():
    source = "dict(a=b, c=d)"
    assert mutate(Context(source=source, mutation_id=MutationID(source, 1, line_number=0))) == ("dict(a=b, cXX=d)", 1)


def test_mutate_dict2():
    source = "dict(a=b, c=d, e=f, g=h)"
    assert mutate(Context(source=source, mutation_id=MutationID(source, 3, line_number=0))) == ("dict(a=b, c=d, e=f, gXX=h)", 1)


def test_performed_mutation_ids():
    source = "dict(a=b, c=d)"
    context = Context(source=source)
    mutate(context)
    # we found two mutation points: mutate "a" and "c"
    assert context.performed_mutation_ids == [MutationID(source, 0, 0), MutationID(source, 1, 0)]


def test_syntax_error():
    with pytest.raises(Exception):
        mutate(Context(source=':!'))

# TODO: this test becomes incorrect with the new mutation_id system, should try to salvage the idea though...
# def test_mutation_index():
#     source = '''
#
# a = b
# b = c + a
# d = 4 - 1
#
#
#     '''.strip()
#     num_mutations = count_mutations(Context(source=source))
#     mutants = [mutate(Context(source=source, mutation_id=i)) for i in range(num_mutations)]
#     assert len(mutants) == len(set(mutants))  # no two mutants should be the same
#
#     # invalid mutation index should not mutate anything
#     mutated_source, count = mutate(Context(source=source, mutation_id=num_mutations + 1))
#     assert mutated_source.strip() == source
#     assert count == 0


def test_bug_github_issue_18():
    source = """@register.simple_tag(name='icon')
def icon(name):
    if name is None:
        return ''
    tpl = '<span class="glyphicon glyphicon-{}"></span>'
    return format_html(tpl, name)"""
    mutate(Context(source=source))


def test_bug_github_issue_19():
    source = """key = lambda a: "foo"
filters = dict((key(field), False) for field in fields)"""
    mutate(Context(source=source))


@pytest.mark.skipif(sys.version_info < (3, 6), reason="Don't check Python 3.6+ syntax in Python < 3.6")
def test_bug_github_issue_26():
    source = """
class ConfigurationOptions(Protocol):
    min_name_length: int
    """
    mutate(Context(source=source))


@pytest.mark.skipif(sys.version_info < (3, 0), reason="Don't check Python 3 syntax in Python 2")
def test_bug_github_issue_30():
    source = """
def from_checker(cls: Type['BaseVisitor'], checker) -> 'BaseVisitor':
    pass
"""
    assert mutate(Context(source=source)) == (source, 0)


def test_bug_github_issue_77():
    # Don't crash on this
    Context(source='')


def test_multiline_dunder_whitelist():
    source = """
__all__ = [
    1,
    2,
]
"""
    assert mutate(Context(source=source)) == (source, 0)

@pytest.mark.parametrize(
    'original, expected', [
        ('try:\n    pass\nfinally:\n    a=4\n',
            ['try:\n    pass\nfinally: pass\n']),

        ('try:\n    pass\nexcept Exception as e:\n    a=4\n',
            ['try:\n    pass\nexcept Exception as e: pass\n',
             'try:\n    pass\nexcept Exception as e: raise\n']),

        ('try:\n    pass\nexcept FooException as e:\n    a=4\nelse:\n    a=5\nfinally:\n    a=6\n',
         ['try:\n    pass\nexcept FooException as e: pass\nelse:\n    a=5\nfinally:\n    a=6\n',
          'try:\n    pass\nexcept FooException as e: raise\nelse:\n    a=5\nfinally:\n    a=6\n',
          'try:\n    pass\nexcept FooException as e:\n    a=4\nelse: pass\nfinally:\n    a=6\n',
          'try:\n    pass\nexcept FooException as e:\n    a=4\nelse:\n    a=5\nfinally: pass\n'
          ]),
    ]
)

def test_try_block_mutations(original, expected):
    """
    Body copied from test_basic_multiple_mutations
    """
    actual = list_mutations(Context(source=original))
    actual_mutation_code = []
    for m in actual:
        mutation, num_mutations = mutate(Context(source=original, mutation_id=m))
        assert num_mutations == 1
        actual_mutation_code.append(mutation)
    for m in expected:
        assert m in actual_mutation_code

def test_mutate_list_comprehension():
    source = 'z = [x for x in y]'
    mutations = list_mutations(Context(source=source))
    assert len(mutations) == 3
    # This can't be tested in basic because the unproductive not in mutation interferes with the [] mutation, and with the cache this clobbers the None mutation. This has been worked around by unmutating the not in.
    assert mutate(Context(source=source, mutation_id=mutations[0])) == ('z = [x for x not in y]', 1)
    assert mutate(Context(source=source, mutation_id=mutations[1])) == ('z = [x for x in []]', 1)
    assert mutate(Context(source=source, mutation_id=mutations[2])) == ('z = None', 1)


