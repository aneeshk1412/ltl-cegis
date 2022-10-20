#include <stdio.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int WALL = 100;
int GOAL = 101;
int check_WALL(int x, int off)
{
    if (x+off >= -10 && x+off <= 10) return 0;
    return 1;
}
int check_GOAL(int x, int off)
{
    return 0;
}
int LEFT = 103;
int RIGHT = 104;
int NONE = 105;
int randint_robot_state_pos(void)
{
    int x = __VERIFIER_nondet_int();
    __VERIFIER_assume((x >= -10 && x <= 10));
    return x;
}
int StateRobotPos;
int StateRobotAct;
void initialize(void)
{
    StateRobotPos = randint_robot_state_pos();
    StateRobotAct = LEFT;
}
void update(void)
{
    if (StateRobotAct == LEFT) StateRobotPos = StateRobotPos - 1;
    if (StateRobotAct == RIGHT) StateRobotPos = StateRobotPos + 1;
    if (StateRobotAct == NONE) StateRobotPos = StateRobotPos;
}
int policy(void)
{
INSERT_ASP}
int check_WALL_StateRobotPos;
void compute_spec_intermediates(void)
{
    check_WALL_StateRobotPos = check_WALL(StateRobotPos, 0)
}
int main(void)
{
    initialize();
    compute_spec_intermediates();
    //@ assert (check_WALL_StateRobotPos == 0);
    while (1) {
        StateRobotAct = policy();

        update();
        compute_spec_intermediates();
        //@ assert (check_WALL_StateRobotPos == 0);
    }
}