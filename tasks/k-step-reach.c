#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

enum robotAction{LEFT, RIGHT, UP, DOWN, NONE};

int StateRobotPosx;
int StateRobotPosy;
enum robotAction StateRobotAct;

void apply_StateRobotAct()
{
    if (StateRobotAct == UP) StateRobotPosy = StateRobotPosy + 1;
    if (StateRobotAct == DOWN) StateRobotPosy = StateRobotPosy - 1;
    if (StateRobotAct == LEFT) StateRobotPosx = StateRobotPosx - 1;
    if (StateRobotAct == RIGHT) StateRobotPosx = StateRobotPosx + 1;
}

int check_prop_WALL(int px, int py)
{
    if ((px >= 1 && px <= 20) && (py >= 1 && py <= 10)) return 0;
    return 1;
}
int check_prop_GOAL(int px, int py)
{
    if ((px == 20 && py == 10)) return 1;
    return 0;
}

int hits_wall = 0;
int at_goal = 0;
int T;

void compute_atomic_propositions(void)
{
    hits_wall = check_prop_WALL(StateRobotPosx, StateRobotPosy);
    at_goal = check_prop_GOAL(StateRobotPosx, StateRobotPosy);
}

enum robotAction policy(void)
{
    if (check_prop_GOAL(StateRobotPosx, StateRobotPosy)) return NONE;
    if (check_prop_WALL(StateRobotPosx, StateRobotPosy+1)) return RIGHT;
    // if (check_prop_WALL(StateRobotPosx+1, StateRobotPosy)) return UP;
    return RIGHT;
}

int main(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPosx >= 1 && StateRobotPosx <= 20)));
    StateRobotPosy = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPosy >= 1 && StateRobotPosy <= 10)));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotAct == LEFT) || (StateRobotAct == RIGHT) || (StateRobotAct == UP) || (StateRobotAct == DOWN)));

    compute_atomic_propositions();
    T = 0;
    //@ assert (hits_wall == 0);
    //@ assert ((T > 20 + 10) ==> (at_goal == 1));
    while (T < 40)
    {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        T++;
        //@ assert (hits_wall == 0);
        //@ assert ((T > 20 + 10) ==> (at_goal == 1));
    }
    return 0;
}
