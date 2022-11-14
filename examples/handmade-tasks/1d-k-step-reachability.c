#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int LEFT = 102;
int RIGHT = 101;
int NONE = 100;

int L = 200;
int T = 0;

int StateRobotPosx;
int StateRobotAct;

void apply_StateRobotAct(void)
{
    if (StateRobotAct == LEFT) StateRobotPosx = StateRobotPosx - 1;
    if (StateRobotAct == RIGHT) StateRobotPosx = StateRobotPosx + 1;
    if (StateRobotAct == NONE) StateRobotPosx = StateRobotPosx;
}

int check_prop_WALL(int px)
{
    if ((px >= 1 && px <= L)) return 0;
    return 1;
}
int check_prop_GOAL(int px)
{
    if ((px == L)) return 1;
    return 0;
}

int hits_WALL = 0;
int reached_GOAL = 0;
void compute_atomic_propositions(void)
{
    hits_WALL = check_prop_WALL(StateRobotPosx);
    reached_GOAL = check_prop_GOAL(StateRobotPosx);
}

int policy(void)
{
    if (StateRobotAct == LEFT) {
        if (!check_prop_GOAL(StateRobotPosx)) return RIGHT;
        if (0) return LEFT;
        return NONE;
    }
    if (StateRobotAct == RIGHT) {
        if (!check_prop_GOAL(StateRobotPosx)) return RIGHT;
        if (0) return LEFT;
        return NONE;
    }
    if (StateRobotAct == NONE) {
        return NONE;
    }
}

int main(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    __VERIFIER_assume(StateRobotPosx >= 1 && StateRobotPosx <= L);
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(StateRobotAct == LEFT || StateRobotAct == RIGHT);

    compute_atomic_propositions();
    //@ loop invariant (hits_WALL == 0);
    while (T <= L)
    {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        //@ assert (T >= L) ==> (reached_GOAL == 1); 
        T++;
    }
    return 0;
}
