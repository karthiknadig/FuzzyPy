import math

class FuzzyFileFormatError(Exception):
    def __init__(self, value):
        self.message = value
    def __str__(self):
        return self.message

def fis_parse_string(fis_string:str):
    """Parses a MATLAB Fuzzy Inference System passed in as a string"""
    lines = fis_string.splitlines()
    def _get_fis_system(lines):
        fisSystem = {}
        if lines[0] == '[System]':
            for line in lines[1:]:
                if len(line)==0 or line.isspace() or line.startswith("["):
                    break
                elif line.startswith('Name'):
                    fisSystem['Name'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('Type'):
                    fisSystem['Type'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('Version'):
                    fisSystem['Version'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('NumInputs'):
                    fisSystem['NumInputs'] = int(line.split('=')[1].strip().strip('"\''))
                    continue
                elif line.startswith('NumOutputs'):
                    fisSystem['NumOutputs'] = int(line.split('=')[1].strip().strip('"\''))
                    continue
                elif line.startswith('NumRules'):
                    fisSystem['NumRules'] = int(line.split('=')[1].strip().strip('"\''))
                    continue
                elif line.startswith('AndMethod'):
                    fisSystem['AndMethod'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('OrMethod'):
                    fisSystem['OrMethod'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('ImpMethod'):
                    fisSystem['ImpMethod'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('AggMethod'):
                    fisSystem['AggMethod'] = line.split('=')[1].strip().strip('"\'')
                    continue
                elif line.startswith('DefuzzMethod'):
                    fisSystem['DefuzzMethod'] = line.split('=')[1].strip().strip('"\'')
                    continue
        else:
            raise FuzzyFileFormatError("FIS file must start with a system block")
        return fisSystem
    def _get_fis_io(lines, isinput=True):
        ios = []
        start = "[Output"
        if isinput:
            start = "[Input"
        for index, ln in enumerate(lines):
            if ln.startswith(start):
                io = {'MF':[]}
                for line in lines[(index+1):]:
                    if len(line)==0 or line.isspace() or line.startswith("["):
                        break
                    elif line.startswith('Name'):
                        io['Name'] = line.split('=')[1].strip().strip('"\'')
                        continue
                    elif line.startswith('Range'):
                        io['Range'] = [float(x) for x in line.split('=')[1].strip().strip('[]').split(' ')]
                        continue
                    elif line.startswith('NumMFs'):
                        io['NumMFs'] = int(line.split('=')[1].strip().strip('"\''))
                        continue
                    elif line.startswith('MF'):
                        mf = {'Index':len(io['MF'])+1}
                        mfpart1 = line.split('=')[1].strip().split(':')
                        mf['Name'] = mfpart1[0].strip('"\'')
                        mfpart2 = mfpart1[1].split(',')
                        mf['Method'] = mfpart2[0].strip('"\'')
                        mf['Args'] = [float(x) for x in mfpart2[1].strip('[]').split(' ')]
                        io['MF'].append(mf);
                ios.append(io)
        if len(ios) <= 0:
            raise FuzzyFileFormatError('FIS file must have at least one Input and one Ouput block')
        return ios
    def _get_fis_rules(lines):
        rules = []
        for index, ln in enumerate(lines):
            if ln == "[Rules]":
                for line in lines[(index+1):]:
                    rule = {'Index':len(rules)+1}
                    if len(line)==0 or line.isspace() or line.startswith("["):
                        break
                    rule_part1 = line.split(',') 
                    rule['Inputs'] = [int(x) for x in rule_part1[0].strip().split(' ')]
                    rule_part2 = rule_part1[1].split('(')
                    rule['Outputs'] = [int(x) for x in rule_part2[0].strip().split(' ')]
                    rule_part3 = rule_part2[1].split(':')
                    rule['Weight'] = float(rule_part3[0].strip().strip(')'))
                    rule_type = int(rule_part3[1].strip())
                    rule['Type'] = 'Or'
                    if rule_type == 1:
                        rule['Type'] = 'And'
                    rules.append(rule)
        if len(rules) <= 0:
            raise FuzzyFileFormatError('FIS file must have one or more rule(s)')
        return rules
    return {
        'System': _get_fis_system(lines),
        'Inputs' : _get_fis_io(lines, True),
        'Outputs' : _get_fis_io(lines, False),
        'Rules' : _get_fis_rules(lines)
        }

def fis_parse_file(fis_file_path:str):
    """Parses a MATLAB Fuzzy Inference System File"""
    contents = None
    with open(fis_file_path, 'rt') as f:
        contents = f.read()
    return fis_parse_string(contents)

def fis_to_string(fis_dict:dict):
    pass

def fis_sigmf(x:float, a:float, c:float):
    """Sigmoid Member Function"""
    return (1.0 / (1.0 + math.exp(-a * (x - c))))
def fis_dsigmf(x:float, a1:float, c1:float, a2:float, c2:float):
    """Double Sigmoid Member Function"""
    return abs(fis_sigmf(x, a1, c1) - fis_sigmf(x, a2, c2))
def fis_gaussmf(x:float, s:float, c:float):
    """Gaussian Member Function"""
    t = (x - c) / s
    return math.exp(-(t * t) / 2)
def fis_gauss2mf(x:float, s1:float, c1:float, s2:float, c2:float):
    """Split Gaussian Member Function"""
    t1 = 1.0
    t2 = 1.0
    if x < c1:
        t1 = fis_gaussmf(x, s1, c1)
    if x > c2:
        t2 = fis_gaussmf(x, s2, c2)
    return (t1 * t2)
def fis_gbellmf(x:float, a:float, b:float, c:float):
    """Generalized Bell Member Function"""
    t = (x - c) / a
    if (t == 0) and (b == 0):
        return 0.5
    if (t == 0) and (b < 0):
        return 0
    return (1.0 / (1.0 + (t ** b)))
def fis_smf(x:float, a:float, b:float):
    """S-Shaped membership function"""
    m = ((a + b) / 2.0)
    t = (b - a)
    if a >= b:
        return float(x >= m)
    if x <= a:
        return 0.0;
    if x <= m:
        t = (x - a) / t
        return (2.0 * t * t)
    if x <= b:
        t = (b - x) / t
        return (1.0 - (2.0 * t * t))
    return 1.0
def fis_zmf(x:float, a:float, b:float):
    """Z-shaped Member Function"""
    m = ((a + b) / 2.0)
    t = (b - a)
    if x <= a:
        return 1.0
    if x <= m:
        t = (x - a) / t
        return (1.0 - (2.0 * t * t))
    if x <= b:
        t = (b - x) / t
        return (1.0 - (2.0 * t * t))
    return 0.0
def fis_trapmf(x:float, a:float, b:float, c:float, d:float):
    """Trapezoidal Member Function"""
    t1 = 0.0
    if x <= c:
        t1 = 1.0
    elif d < x:
        t1 = 0.0
    elif not c == d:
        t1 = (d - x) / (d - c)
    else:
        t1 = 0.0
    t2 = 0.0
    if b <= x:
        t2 = 1.0
    elif x < a:
        t2 = 0.0
    elif not a == b:
        t2 = (x - a) / (b - a)
    else:
        t2 = 0.0
    return min(t1, t2)
def fis_pimf(x:float, a1:float, b1:float, a2:float, b2:float):
    """Pi-shaped Member Function"""
    return fis_smf(x, a1, b1) * fis_zmf(x, a2, b2)
def fis_psigmf(x:float, a1:float, c1:float, a2:float, c2:float):
    """Product of Sigmoid Member Function"""
    return (fis_sigmf(x, a1, c1) * fis_sigmf(x, a2, c2))
def fis_trimf(x:float, a:float, b:float, c:float):
    """Triangular Member Function"""
    t1 = (x - a) / (b - a)
    t2 = (c - x) / (c - b)
    if (a == b) and (b == c):
        return float(x == a)
    if (a == b):
        return (t2 * float(b <= x) * float(x <= c))
    if (b == c):
        return (t1 * float(a <= x) * float(x <= b))
    return max(min(t1, t2), 0)
    
