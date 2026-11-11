"""Restricted arithmetic reader for native input parameters and regression checks.

Only arithmetic, arrays and SIN/COS/TAN are accepted. OPALX evaluates the actual
course inputs. The notebook uses these parameters for labels, controls and checks;
closed orbits and full tunes are always computed by OPALX.
"""
import ast
import math
import re


def evaluate(expression, constants):
    expression=expression.strip()
    if expression.startswith('{'):
        return [evaluate(value,constants) for value in expression.strip('{}').split(',')]
    if expression.startswith('"'):return ast.literal_eval(expression)
    functions={'SIN':math.sin,'COS':math.cos,'TAN':math.tan}
    def visit(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return node.value
        if isinstance(node,ast.Name):return constants[node.id]
        if isinstance(node,ast.UnaryOp):
            value=visit(node.operand)
            if isinstance(node.op,ast.USub):return -value
            if isinstance(node.op,ast.UAdd):return value
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow):return a**b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            return functions[node.func.id.upper()](*(visit(arg) for arg in node.args))
        raise ValueError(f'Unsupported native arithmetic: {expression}')
    return visit(ast.parse(expression.replace('^','**'),mode='eval').body)


def evaluated_input(text):
    text=re.sub(r'//[^\n]*','',text)
    constants={'PI':math.pi}
    for name,value in re.findall(r'\bREAL\s+(\w+)\s*=\s*([^;]+);',text):
        constants[name]=evaluate(value,constants)
    definitions={}
    for name,kind,body in re.findall(r'(\w+)\s*:\s*(ASYMMETRIC_ENGE|SCALINGFFAMAGNET|BEAM|COF)\s*,([^;]+);',text,re.S):
        attributes={}
        for key,value in re.findall(r'(\w+)\s*=\s*({[^}]*}|[^,;]+)',body):
            try:attributes[key]=evaluate(value,constants)
            except KeyError:attributes[key]=value.strip()
        definitions[name]=(kind,attributes)
    ring=re.search(r'SCHOOL\s*:\s*RING\s*=\s*\(([^)]+)\)',text,re.S)
    members=[member.strip() for member in ring[1].split(',')] if ring else []
    return constants,definitions,members
