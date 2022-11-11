#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int JUMP = 102;
int STEP = 101;
int NONE = 100;

int StateRobotPos;
int StateRobotAct;

void update_StateRobotPos_from_action(int act)
{
    if (act == JUMP) StateRobotPos = StateRobotPos + 2;
    if (act == STEP) StateRobotPos = StateRobotPos + 1;
    if (act == NONE) StateRobotPos = StateRobotPos;
}

int check_prop_WALL(int px)
{
    if (px >= 1 && px <= 10) return 0;
    return 1;
}
int check_prop_HOLE(int px)
{
    if (px == 6 || px == 9) return 1;
    return 0;
}
int check_prop_GOAL(int px)
{
    if (px == 10) return 1;
    return 0;
}

int hits_wall = 0;
int hits_hole = 0;

void compute_atomic_propositions(void)
{
    hits_wall = check_prop_WALL(StateRobotPos);
    hits_hole = check_prop_HOLE(StateRobotPos);
}

int policy(void)
{
    if (StateRobotAct == JUMP) {
        if (!check_prop_GOAL(StateRobotPos)) return STEP;
        return NONE;
    }
    if (StateRobotAct == STEP) {
        if (!check_prop_GOAL(StateRobotPos)) return STEP;
        return NONE;
    }
    if (StateRobotAct == NONE) {
        return NONE;
    }
}

int main(void)
{
    StateRobotPos = __VERIFIER_nondet_int();
    __VERIFIER_assume((StateRobotPos != 6 && StateRobotPos != 9 && StateRobotPos >= 1 && StateRobotPos <= 10));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotAct == JUMP) || (StateRobotAct == STEP)));

    compute_atomic_propositions();
    //@ assert (hits_wall == 0);
    //@ assert (hits_hole == 0);
    while (1)
    {
        StateRobotAct = policy();
        update_StateRobotPos_from_action(StateRobotAct);
        compute_atomic_propositions();
        //@ assert (hits_wall == 0);
        //@ assert (hits_hole == 0);
    }
    return 0;
}
