#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int vector_add(int x, int d)
{
    return x + d;
}

int LEFT = 100;
int RIGHT = 101;

int update_pos_from_action(int act, int x)
{
    if (act == LEFT) return x - 1;
    if (act == RIGHT) return x + 1;
}

int check_prop_WALL(int x)
{
    if ((x >= -10 && x <= 10)) return 0;
    return 1;
}
int check_prop_CHECKPOINT(int x)
{
    if ((x == 0)) return 1;
    return 0;
}

int StateRobotPos;
int StateRobotAct;

void initialize(void)
{
    StateRobotPos = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPos >= -10 && StateRobotPos <= 10)));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotAct == LEFT) || (StateRobotAct == RIGHT)));
}

int policy(void)
{
    INSERT_ASP
    return RIGHT;
}

int do_not_hit_wall;
void compute_spec(void)
{
    do_not_hit_wall = check_prop_WALL(StateRobotPos) == 0;
}

int main(void)
{
    initialize();
    compute_spec();
    //@ assert (do_not_hit_wall != 0);
    while (1) {
        StateRobotAct = policy();
        StateRobotPos = update_pos_from_action(StateRobotAct, StateRobotPos);
        compute_spec();
        //@ assert (do_not_hit_wall != 0);
    }
}
