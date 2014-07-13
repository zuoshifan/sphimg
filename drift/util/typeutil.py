
def positive_int(val):
    try:
        val = int(val)
        if val > 0:
            return val
        else:
            raise ValueError('')
    except Exception:
        raise ValueError('Require a positive interger.')

def natural_int(val):
    try:
        val = int(val)
        if val >= 0:
            return val
        else:
            raise ValueError('')
    except Exception:
        raise ValueError('Require a natural number.')

def none_or_int(val):
    try:
        if val is not None:
            val = int(val)
        return val
    except Exception:
        raise ValueError('Require None or a integer.')

def none_or_positive_int(val):
    try:
        if val is not None:
            val = positive_int(val)
        return val
    except ValueError:
        raise ValueError('Require None or a positive interger.')

def none_or_natural_int(val):
    try:
        if val is not None:
            val = natural_int(val)
        return val
    except ValueError:
        raise ValueError('Require None or a natural number.')

def positive_float(val):
    try:
        val = float(val)
        if val > 0.0:
            return val
        else:
            raise ValueError('')
    except Exception:
        raise ValueError('Require a positive float number.')

def nonnegative_float(val):
    try:
        val = float(val)
        if val >= 0.0:
            return val
        else:
            raise ValueError('')
    except Exception:
        raise ValueError('Require a non-negative float number.')

def none_or_float(val):
    try:
        if val is not None:
            val = float(val)
        return val
    except Exception:
        raise ValueError('Require None or a float number.')

def none_or_positive_float(val):
    try:
        if val is not None:
            val = positive_float(val)
        return val
    except ValueError:
        raise ValueError('Require None or a positive float number.')

def none_or_nonnegative_float(val):
    try:
        if val is not None:
            val = nonnegative_float(val)
        return val
    except ValueError:
        raise ValueError('Require None or a non-negative float number.')

def ispower(n, base):
    if n == base:
        return True
    if base == 1:
        return False
    temp = base
    while (temp <= n):
        if temp == n:
            return True
        temp *= base
    return False

def power_of_2(val):
    try:
        val = positive_int(val)
        if ispower(val, 2):
            return val
        else:
            raise ValueError('')
    except ValueError:
        raise ValueError('Require a positive integer which is the power of 2.')

def non_empty_list(lst):
    try:
        lst = [val for val in lst]
        if lst:
            return lst
        else:
            raise ValueError('')
    except Exception:
        raise ValueError('Require a non-empty list.')

def int_or_list(lst):
    try:
        try:
            lst = [int(val) for val in lst]
            return lst
        except Exception:
            lst = [int(lst)]
            return lst
    except Exception:
        raise ValueError('Require a integer or a list of integers.')

def positive_int_or_list(lst):
    try:
        try:
            lst = [positive_int(val) for val in lst]
            return lst
        except Exception:
            lst = [positive_int(lst)]
            return lst
    except ValueError:
        raise ValueError('Require a positive integer or a list of positive integers.')

def natural_int_or_list(lst):
    try:
        try:
            lst = [natural_int(val) for val in lst]
            return lst
        except Exception:
            lst = [natural_int(lst)]
            return lst
    except ValueError:
        raise ValueError('Require a natural number or a list of natural numbers.')

def int_or_non_empty_list(lst):
    try:
        return non_empty_list(int_or_list(lst))
    except ValueError:
        raise ValueError('Require a integer or a non-empty list of integers.')

def positive_int_or_non_empty_list(lst):
    try:
        return non_empty_list(positive_int_or_list(lst))
    except ValueError:
        raise ValueError('Require a positive integer or a non-empty list of positive integers.')

def natural_int_or_non_empty_list(lst):
    try:
        return non_empty_list(natural_int_or_list(lst))
    except ValueError:
        raise ValueError('Require a natural number or a non-empty list of natural numbers.')

def float_or_list(lst):
    try:
        try:
            lst = [float(val) for val in lst]
            return lst
        except Exception:
            lst = [float(lst)]
            return lst
    except Exception:
        raise ValueError('Require a float number or a list of float nubers.')

def positive_float_or_list(lst):
    try:
        try:
            lst = [positive_float(val) for val in lst]
            return lst
        except Exception:
            lst = [positive_float(lst)]
            return lst
    except ValueError:
        raise ValueError('Require a positive float number or a list of positive float numbers.')

def nonnegative_float_or_list(lst):
    try:
        try:
            lst = [nonnegative_float(val) for val in lst]
            return lst
        except Exception:
            lst = [nonnegative_float(lst)]
            return lst
    except ValueError:
        raise ValueError('Require a non-negative float number or a list of non-negative float numbers.')

def float_or_non_empty_list(lst):
    try:
        return non_empty_list(float_or_list(lst))
    except ValueError:
        raise ValueError('Require a float number or a non-empty list of float numbers.')

def positive_float_or_non_empty_list(lst):
    try:
        return non_empty_list(positive_float_or_list(lst))
    except ValueError:
        raise ValueError('Require a positive float number or a non-empty list of positive float numbers.')

def nonnegative_float_or_non_empty_list(lst):
    try:
        return non_empty_list(nonnegative_float_or_list(lst))
    except ValueError:
        raise ValueError('Require a non-negative float number or a non-empty list of non-negative float numbers.')

