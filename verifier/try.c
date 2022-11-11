//@ ltl invariant positive: []<>(AP(1 == 1));

#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int JUMP = 100;
int STEP = 101;
int NONE = 102;

int StateRobotPosx;
int StateRobotAct;

void apply_StateRobotAct(void)
{
    if (StateRobotAct == JUMP) StateRobotPosx = StateRobotPosx + 2;
    if (StateRobotAct == STEP) StateRobotPosx = StateRobotPosx + 1;
    if (StateRobotAct == NONE) StateRobotPosx = StateRobotPosx;
}

int check_prop_WALL(int px)
{
    if ((px >= 0 && px <= 20)) return 0;
    return 1;
}
int check_prop_HOLE(int px)
{
    if ((px == 3)) return 1;
    if ((px == 10)) return 1;
    if ((px == 14)) return 1;
    if ((px == 17)) return 1;
    return 0;
}
int check_prop_GOAL(int px)
{
    if ((px == 20)) return 1;
    return 0;
}

int HitsHole = 0;
int ReachesGoal = 0;

void compute_atomic_propositions(void)
{
    HitsHole = check_prop_HOLE(StateRobotPosx) == 1;
    ReachesGoal = check_prop_GOAL(StateRobotPosx) == 1;
}

int policy(void)
{
    if (check_prop_GOAL(StateRobotPosx)) return NONE;
    if (check_prop_HOLE(StateRobotPosx + 1)) return JUMP;
    return STEP;
}

void initialize(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    __VERIFIER_assume((((StateRobotPosx >= 0 && StateRobotPosx <= 2)) || ((StateRobotPosx >= 4 && StateRobotPosx <= 9)) || ((StateRobotPosx >= 11 && StateRobotPosx <= 13)) || ((StateRobotPosx >= 15 && StateRobotPosx <= 16)) || ((StateRobotPosx >= 18 && StateRobotPosx <= 19))));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume((StateRobotAct == JUMP || StateRobotAct == STEP || StateRobotAct == NONE));
}

int main(void)
{
    initialize();
    compute_atomic_propositions();
    //@ assert (HitsHole == 0);
    while (1)
    {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        //@ assert (HitsHole == 0);
        printf("StateRobotPosx: %d, StateRobotAct: %d, HitsHole: %d, ReachesGoal: %d\n", StateRobotPosx, StateRobotAct, HitsHole, ReachesGoal);
    }
}
