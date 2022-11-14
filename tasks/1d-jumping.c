#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

// PARAMETERS
int HALL_LENGTH = 20;
int NUM_HOLES = 5;
int HOLES[] = {5, 7, 10, 14, 18};
int T = 0;
int T_LIMIT = HALL_LENGTH;

int JUMP = 102;
int STEP = 101;
int NONE = 100;

int StateRobotPosx;
int StateRobotAct;

void apply_StateRobotAct()
{
    if (StateRobotAct == NONE) StateRobotPosx = StateRobotPosx;
    if (StateRobotAct == STEP) StateRobotPosx = StateRobotPosx + 1;
    if (StateRobotAct == JUMP) StateRobotPosx = StateRobotPosx + 2;
}

int check_WALL(int px)
{
    if (px >= 1 && px <= HALL_LENGTH) return 0;
    return 1;
}
int check_GOAL(int px)
{
    if (px == HALL_LENGTH) return 1;
    return 0;
}
int check_HOLE(int px)
{
    for (int i=0; i<NUM_HOLES; i++)
        if (px == HOLES[i]) return 1;
    return 0;
}

void initialize(void)
{
    StateRobotPosx = 1;
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(StateRobotAct == STEP || StateRobotAct == JUMP || StateRobotAct == NONE);
}

int policy(void)
{
    INSERT_ASP
}

int hits_WALL;
int hits_HOLE;
int reaches_GOAL;
void compute_atomic_propositions(void)
{
    hits_WALL = check_WALL(StateRobotPosx) == 1;
    hits_HOLE = check_HOLE(StateRobotPosx) == 1;
    reaches_GOAL = check_GOAL(StateRobotPosx) == 1;
}

int main(void)
{
    initialize();
    compute_atomic_propositions();
    //@ loop invariant (hits_WALL == 0);
    //@ loop invariant (hits_HOLE == 0);
    while (T <= T_LIMIT) {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        T++;
        //@ assert (T >= T_LIMIT) ==> (reaches_GOAL == 1);
    }
}

/*
SAT ASP:
    if (check_GOAL(StateRobotPosx)) return NONE;
    if (check_HOLE(StateRobotPosx + 1)) return JUMP;
    if (1) return STEP;
*/
