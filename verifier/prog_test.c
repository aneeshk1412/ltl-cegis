//@ ltl invariant positive: []<>(AP(reaches_checkpoint1 == 1)) && []<>(AP(reaches_checkpoint2 == 1));

#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int NONE = 100;
int LEFT = 101;
int RIGHT = 102;

int StateRobotPosx;
int StateRobotAct;

void update_StateRobotPos_from_action(int act)
{
    if (act == LEFT) StateRobotPosx = StateRobotPosx - 1;
    if (act == RIGHT) StateRobotPosx = StateRobotPosx + 1;
}

int check_prop_WALL(int x)
{
    if ((x >= -10 && x <= 10)) return 0;
    return 1;
}
int check_prop_CHECKPOINT1(int x)
{
    if ((x == -5)) return 1;
    return 0;
}
int check_prop_CHECKPOINT2(int x)
{
    if ((x == 5)) return 1;
    return 0;
}

void initialize(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPosx >= -10 && StateRobotPosx <= 10)));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotAct == LEFT) || (StateRobotAct == RIGHT)));
}

int policy(void)
{
    if ((StateRobotAct == RIGHT)) return LEFT;
    if ((StateRobotAct == LEFT)) return RIGHT;
    return StateRobotAct;
}

int hits_wall = 0;
int reaches_checkpoint1 = 0;
int reaches_checkpoint2 = 0;
void compute_atomic_propositions(void)
{
    hits_wall = check_prop_WALL(StateRobotPosx) == 1;
    reaches_checkpoint1 = check_prop_CHECKPOINT1(StateRobotPosx) == 1;
    reaches_checkpoint2 = check_prop_CHECKPOINT2(StateRobotPosx) == 1;
}

int main(void)
{
    initialize();
    compute_atomic_propositions();
    if (hits_wall) {
        __VERIFIER_error();
    }
    while (1) {
        StateRobotAct = policy();
        update_StateRobotPos_from_action(StateRobotAct);
        compute_atomic_propositions();
        if (hits_wall) {
            __VERIFIER_error();
        }
    }
}
