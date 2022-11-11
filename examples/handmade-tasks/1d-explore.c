//@ ltl invariant positive: []AP(temp == 0);
#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int temp = 0;

int LEFT = 102;
int RIGHT = 101;
int NONE = 100;

int StateRobotPosx;
int StateRobotAct;
int IsButtonPressed;

void update_StateRobotPos_from_action(int act)
{
    if (act == LEFT) StateRobotPosx = StateRobotPosx - 1;
    if (act == RIGHT) StateRobotPosx = StateRobotPosx + 1;
    if (act == NONE) StateRobotPosx = StateRobotPosx;
}

int check_prop_WALL(int px)
{
    if ((px >= 1 && px <= 20) ) return 0;
    return 1;
}
int check_prop_GOAL(int px)
{
    if ((px == 20)) return 1;
    return 0;
}
int check_prop_BUTTON(int px)
{
    if ((px == 10)) return 1;
    return 0;
}

int hits_wall = 0;

void compute_atomic_propositions(void)
{
    hits_wall = check_prop_WALL(StateRobotPosx);
    if (IsButtonPressed == 0) {
        IsButtonPressed = check_prop_BUTTON(StateRobotPosx);
    }
}

int policy(void)
{
    if (StateRobotAct == LEFT) {
        if (check_prop_WALL(StateRobotPosx - 1)) return RIGHT;
        if (1) return LEFT;
        return NONE;
    }
    if (StateRobotAct == RIGHT) {
        if (check_prop_GOAL(StateRobotPosx + 1) && IsButtonPressed) return RIGHT;
        if (check_prop_GOAL(StateRobotPosx + 1)) return LEFT;
        return NONE;
    }
    if (StateRobotAct == NONE) {
        return NONE;
    }
}

int main(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPosx >= 1 && StateRobotPosx <= 19)));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotAct == LEFT) || (StateRobotAct == RIGHT)));
    IsButtonPressed = 0;

    compute_atomic_propositions();
    // if (hits_wall) __VERIFIER_error();
    //@ assert (hits_wall == 0);
    int t=0;
    while (t < 3 * 20)
    {
        StateRobotAct = policy();
        update_StateRobotPos_from_action(StateRobotAct);
        compute_atomic_propositions();
        // if (hits_wall) __VERIFIER_error();
        //@ assert (hits_wall == 0);
        // if (StateRobotAct == NONE) return 0;
        t++;
    }
    return 0;
}
