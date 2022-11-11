#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

typedef struct {
    int x;
    int y;
} Vector;

int UP = 104;
int DOWN = 103;
int LEFT = 102;
int RIGHT = 101;
int NONE = 100;

Vector StateRobotPos;
int StateRobotAct;

Vector vector_add(Vector p, int dx, int dy) {
    p.x += dx;
    p.y += dy;
}

void apply_StateRobotAct()
{
    if (StateRobotAct == UP) StateRobotPos.y = StateRobotPos.y + 1;
    if (StateRobotAct == DOWN) StateRobotPos.y = StateRobotPos.y - 1;
    if (StateRobotAct == LEFT) StateRobotPos.x = StateRobotPos.x - 1;
    if (StateRobotAct == RIGHT) StateRobotPos.x = StateRobotPos.x + 1;
}

int check_prop_WALL(Vector p)
{
    if ((p.x >= 1 && p.x <= 20) && (p.y >= 1 && p.y <= 10)) return 0;
    return 1;
}
int check_prop_GOAL(Vector p)
{
    if ((p.x == 20 && p.y == 10)) return 1;
    return 0;
}

int hits_wall = 0;
int at_goal = 0;
int T;

void compute_atomic_propositions(void)
{
    hits_wall = check_prop_WALL(StateRobotPos);
    at_goal = check_prop_GOAL(StateRobotPos);
}

int policy(void)
{
    if (check_prop_GOAL(StateRobotPos)) return NONE;
    if (check_prop_WALL(vector_add(StateRobotPos, 0, 1))) return RIGHT;
    if (check_prop_WALL(vector_add(StateRobotPos, 1, 0))) return UP;
    return RIGHT;
}

int main(void)
{
    StateRobotPos.x = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPos.x >= 1 && StateRobotPos.x <= 20)));
    StateRobotPos.y = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotPos.y >= 1 && StateRobotPos.y <= 10)));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume(((StateRobotAct == LEFT) || (StateRobotAct == RIGHT) || (StateRobotAct == UP) || (StateRobotAct == DOWN)));

    compute_atomic_propositions();
    T = 0;
    //@ assert (hits_wall == 0);
    //@ assert ((T > 20 + 10) ==> (at_goal == 1));
    while (1)
    {
        StateRobotAct = policy();
        update_StateRobotPos_from_action(StateRobotAct);
        compute_atomic_propositions();
        T++;
        //@ assert (hits_wall == 0);
        //@ assert ((T > 20 + 10) ==> (at_goal == 1));
    }
    return 0;
}
