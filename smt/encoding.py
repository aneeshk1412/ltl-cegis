
from z3 import *




def create_context(check_wall, offset):
    s = Solver()
    # constraints for the wall
    x = Int('x')
    s.add(ForAll([x], Implies(x>=500, check_wall(x))))
    s.add(ForAll([x], Implies(x<=-500, check_wall(x))))
    s.add(ForAll([x], Implies(And(x<500, x>-500), Not(check_wall(x)))))

    # constratins for the offset
    s.add(And(offset>=-500, offset<=500))

    return s


def complete_sketch(sketch, pos_samples, neg_samples):
    
    #print ('positive samples: '+ str(pos_samples))
    #print ('negative samples: '+ str(neg_samples))
    #print ('current sketch: ' + str(sketch))

    check_wall = Function('check_wall', IntSort(), BoolSort())
    offset = Int('O')
    s = create_context(check_wall, offset)

    # hardcoded for now <TODO>
    if sketch.terms[0] == 'not':
        for poss in pos_samples:
            s.add(Not(check_wall(poss+offset)))
        for negs in neg_samples:
            s.add(check_wall(negs+offset))
    else:
        for poss in pos_samples:
            s.add(check_wall(poss+offset))
        for negs in neg_samples:
            s.add(Not(check_wall(negs+offset)))
    # run the solver
    if s.check() == sat:
        m = s.model()
        #print('offset = '+str(m[offset]))
        return str(sketch).replace('?', str(m[offset]))
    else:
        #print ('no model exists')
        return '??'

    
   


   

    

    
    return None