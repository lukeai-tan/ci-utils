"""Observe original dictionary identities without replacing their values."""

import ast
import importlib.abc
import importlib.machinery
import types
from pathlib import Path

from compiledcheck import children, identity
from fieldcheck import CORE


class Fields:
    def __init__(self, mask=lambda: 1):
        self.mask = mask
        self.objects = {}
        self.fields = {}

    def register(self, value, key):
        if type(value) is dict:
            existing = self.objects.get(id(value))
            if existing:
                # Shared construction is an alias: retain every origin.
                existing[1].add(key)
            else:
                self.objects[id(value)] = (value, {key})
            self.fields.setdefault(key, {'whole': 0, 'keys': {}, 'missing': {}})
        return value

    def mark(self, value, key=None, missing=False):
        entry = self.objects.get(id(value))
        if entry is None:
            return
        for origin in entry[1]:
            row = self.fields[origin]
            if key is None:
                row['whole'] |= self.mask()
            elif type(key) is str:
                kind = 'missing' if missing else 'keys'
                row[kind][key] = row[kind].get(key, 0) | self.mask()
            else:
                row['whole'] |= self.mask()

    def attribute(self, value, name, deferred=False):
        result = getattr(value, name)
        if not deferred:
            self.mark(result)
            self.namespace(value, name)
        return result

    def namespace(self, value, name):
        if name in {'required_args', '__dict__'}:
            cls = value if isinstance(value, type) else type(value)
            if type(cls) is type:
                for base in type.__getattribute__(cls, '__mro__'):
                    namespace = type.__getattribute__(base, '__dict__')
                    for attr, mapping in namespace.items():
                        if name == '__dict__' or attr == 'arg_types':
                            self.mark(mapping)

    def dynamic(self, function, /, *args, **kwargs):
        result = function(*args, **kwargs)
        self.mark(result)
        if args and (function is vars or function is getattr and len(args) > 1
                     and args[1] in {'__dict__', 'required_args'}):
            self.namespace(args[0], '__dict__' if function is vars else args[1])
        return result

    def item(self, mapping, key):
        self.mark(mapping, key)
        return mapping[key]

    def call(self, role, function, /, *args, **kwargs):
        if type(function) is types.BuiltinMethodType:
            mapping = function.__self__
            name = function.__name__
            if name == 'get' and args and len(args) <= 2 and not kwargs:
                self.mark(mapping, args[0])
            elif name == 'items' and role == 'requirements-init' and not args and not kwargs:
                # This exact comprehension derives required_args; its later
                # consumers and missing inputs are recorded separately.
                pass
            elif name in {'keys', '__len__', '__contains__'}:
                pass  # Value edits preserve membership, size and order.
            else:
                self.mark(mapping)
        return function(*args, **kwargs)

    def validate(self, value):
        cls = type(value)
        if type(cls) is not type:
            return
        mapping = None
        for base in type.__getattribute__(cls, '__mro__'):
            namespace = type.__getattribute__(base, '__dict__')
            if mapping is None:
                mapping = namespace.get('arg_types')
        if type(mapping) is not dict or id(mapping) not in self.objects:
            return
        if any('__getattribute__' in type.__getattribute__(base, '__dict__') and base is not object
               for base in type.__getattribute__(cls, '__mro__')):
            self.mark(mapping)
            return
        try:
            args = object.__getattribute__(value, 'args')
            required = object.__getattribute__(value, 'required_args')
            if (type(args) is not dict or type(required) is not set
                    or any(type(k) is not str for k in [*args, *mapping])
                    or any(type(v) is not bool for v in mapping.values())
                    or required != {k for k, v in mapping.items() if v}
                    or any(isinstance(v, list) and type(v) is not list for v in args.values())):
                self.mark(mapping)
                return
            for key in mapping:
                v = args.get(key)
                if v is None or type(v) is list and not v:
                    self.mark(mapping, key, missing=True)
                    if mapping[key]:
                        # Rebuilding a set can reorder other validation errors.
                        self.mark(mapping)
        except (AttributeError, TypeError):
            self.mark(mapping)

    def data(self):
        return {key: {'whole': format(row['whole'], 'x'),
                      'keys': {k: format(v, 'x') for k, v in row['keys'].items()},
                      'missing': {k: format(v, 'x') for k, v in row['missing'].items()}}
                for key, row in self.fields.items()}


class Instrument(ast.NodeTransformer):
    def __init__(self, path, names):
        self.path = path
        self.names = names | {'__dict__', 'required_args'}
        self.owners = []
        self.function = ''

    def hook(self, name, arguments, node):
        return ast.copy_location(ast.Call(ast.Name('_ci_field_' + name, ast.Load()), arguments, []), node)

    def visit_ClassDef(self, node):
        self.owners.append(node.name)
        result = self.generic_visit(node)
        self.owners.pop()
        return result

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = '.'.join([*self.owners, node.name])
        node = self.generic_visit(node)
        if self.path == CORE and self.function == 'Expr.error_messages':
            entry = ast.Expr(self.hook('validate', [ast.Name('self', ast.Load())], node.body[0]))
            node.body.insert(0, ast.copy_location(entry, node.body[0]))
        self.function = previous
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (self.owners and not self.function and len(targets) == 1 and isinstance(targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)):
            key = f'{self.path}:{".".join(self.owners)}:{targets[0].id}'
            node.value = self.hook('register', [self.visit(node.value), ast.Constant(key)], node.value)
            return node
        return self.generic_visit(node)

    visit_AnnAssign = visit_Assign

    def attribute(self, node, deferred=False):
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        # These reads belong to the complete validator proof, including shape
        # checks that are unaffected by Boolean value-only edits.
        deferred |= self.path == CORE and self.function == 'Expr.error_messages' and node.attr in {'arg_types', 'required_args'}
        return self.hook('attribute', [self.visit(node.value), ast.Constant(node.attr), ast.Constant(deferred)], node)

    def visit_Attribute(self, node):
        return self.attribute(node) if node.attr in self.names else self.generic_visit(node)

    def receiver(self, node):
        return self.attribute(node, True) if isinstance(node, ast.Attribute) and node.attr in self.names else self.visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Attribute) and node.value.attr in self.names:
            return self.hook('item', [self.receiver(node.value), self.visit(node.slice)], node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in {'getattr', 'vars'}:
            return ast.copy_location(ast.Call(ast.Name('_ci_field_dynamic', ast.Load()),
                [self.visit(node.func), *[self.visit(arg) for arg in node.args]],
                [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords]), node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr in {'get', 'items', 'keys', 'values', 'update', 'pop', 'setdefault', 'clear', 'copy'}
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr in self.names):
            function = ast.Attribute(self.receiver(node.func.value), node.func.attr, ast.Load())
            role = ('requirements-init' if self.path == CORE and self.function == 'Expr.__init_subclass__'
                    and node.func.attr == 'items' and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == 'arg_types' else '')
            return ast.copy_location(ast.Call(ast.Name('_ci_field_call', ast.Load()),
                [ast.Constant(role), function, *[self.visit(arg) for arg in node.args]],
                [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords]), node)
        return self.generic_visit(node)


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, recorder):
        super().__init__(fullname, path)
        self.recorder = recorder

    def get_code(self, fullname):
        source = self.get_data(self.path).decode('utf-8')
        path = Path(self.path).resolve().relative_to(self.recorder.root).as_posix()
        original = compile(source, self.path, 'exec', dont_inherit=True)
        def register(code):
            self.recorder.original_codes.setdefault((path, code.co_firstlineno, code.co_qualname), []).append(identity(code))
            for child in children(code):
                register(child)
        register(original)
        tree = Instrument(path, self.recorder.field_names).visit(ast.parse(source))
        return compile(ast.fix_missing_locations(tree), self.path, 'exec', dont_inherit=True)

    def exec_module(self, module):
        fields = self.recorder.fields
        module.__dict__.update({f'_ci_field_{name}': getattr(fields, name)
                               for name in ('register', 'attribute', 'item', 'call', 'validate', 'dynamic')})
        super().exec_module(module)
        if module.__name__ == 'sqlglot.executor':
            original = module.optimize
            def optimize(*args, **kwargs):
                return self.recorder.optimize(original, *args, **kwargs)
            module.optimize = optimize


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, recorder):
        self.recorder = recorder

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'sqlglot' and not fullname.startswith('sqlglot.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.origin and spec.origin.endswith('.py'):
            if not Path(spec.origin).resolve().is_relative_to(self.recorder.root):
                raise ValueError('source imported outside pinned root')
            spec.loader = Loader(fullname, spec.origin, self.recorder)
        return spec
