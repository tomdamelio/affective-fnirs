
keywords = ['VERIFICATION', 'CONFIRMED', 'DEBUG', 'Embedded']

def check(encoding):
    try:
        with open('analysis.log', 'r', encoding=encoding) as f:
            found = False
            for line in f:
                if any(k in line for k in keywords):
                    print(line.strip())
                    found = True
            return found
    except Exception:
        return False

# Try encodings
if not check('utf-8'):
    if not check('utf-16'):
        check('latin-1')
