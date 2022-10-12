#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));
int check_TILE(int x_0)
{
    if ((x_0 >= -10 && x_0 <= 10)) return 1;
    return 0;
}
int check_CHECKPOINT(int x_0)
{
    if ((x_0 == 0)) return 1;
    return 0;
}
int check_WALL(int x_0)
{
    if ((x_0 >= -10 && x_0 <= 10)) return 0;
    return 1;
}
int LEFT = 100;
int RIGHT = 101;
int NONE = 102;
int randint_robot_state_pos(void)
{
    int x = __VERIFIER_nondet_int();
    __VERIFIER_assume(((x >= 5 && x <= 10)));
    return x;
}
int robot_state_pos;
int robot_state_act;
void initialize(void)
{
    robot_state_pos = randint_robot_state_pos();
    robot_state_act = LEFT;
}
void policy(void)
{
    if (robot_state_act == LEFT) { robot_state_act = RIGHT; return; }
    if (robot_state_act == RIGHT) { robot_state_act = LEFT; return; }
    robot_state_act = LEFT;
}
void update(void)
{
    if (robot_state_act == LEFT) robot_state_pos = robot_state_pos - 1;
    if (robot_state_act == RIGHT) robot_state_pos = robot_state_pos + 1;
    if (robot_state_act == NONE) robot_state_pos = robot_state_pos;
}
int check_wall_robot_state_pos;
void precompute(void)
{
    check_wall_robot_state_pos = check_WALL(robot_state_pos);
}
int main(void)
{
    initialize();
    precompute();
    while (1) {
        policy();
        update();
        precompute();
        //@ assert (check_wall_robot_state_pos == 0);
    }
}