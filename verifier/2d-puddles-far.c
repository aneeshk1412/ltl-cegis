//@ ltl invariant positive: []<>(AP(1 == 1));

#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int UP = 100;
int DOWN = 101;
int LEFT = 102;
int RIGHT = 103;
int NONE = 104;

int StateRobotPosx;
int StateRobotPosy;
int StateRobotAct;

void apply_StateRobotAct(void)
{
    if (StateRobotAct == UP) StateRobotPosy = StateRobotPosy + 1;
    if (StateRobotAct == DOWN) StateRobotPosy = StateRobotPosy - 1;
    if (StateRobotAct == LEFT) StateRobotPosx = StateRobotPosx - 1;
    if (StateRobotAct == RIGHT) StateRobotPosx = StateRobotPosx + 1;
    if (StateRobotAct == NONE) StateRobotPosx = StateRobotPosx;
}

int check_prop_WALL(int px, int py)
{
    if ((px >= -10 && px <= 5) && (py >= 1 && py <= 6)) return 0;
    return 1;
}
int check_prop_GOAL(int px, int py)
{
    if ((px == 5) && (py == 1)) return 0;
    return 1;
}
int check_prop_HOLE(int px, int py)
{
    if ((px == 3) && (py == 6)) return 1;
    if ((px == 4) && (py == 5)) return 1;
    if ((px == 3) && (py == 4)) return 1;
    if ((px == 3) && (py == 2)) return 1;
    if ((px == 3) && (py == 1)) return 1;
    return 0;
}

int HitsWall = 0;
int HitsHole = 0;
int ReachesGoal = 0;

void compute_atomic_propositions(void)
{
    HitsWall = check_prop_WALL(StateRobotPosx, StateRobotPosy) == 1;
    HitsHole = check_prop_HOLE(StateRobotPosx, StateRobotPosy) == 1;
    ReachesGoal = check_prop_GOAL(StateRobotPosx, StateRobotPosy) == 1;
}

int policy(void)
{
    if (check_prop_HOLE(StateRobotPosx+1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx+1, StateRobotPosy)) return DOWN; 
    return RIGHT;
}

void initialize(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    StateRobotPosy = __VERIFIER_nondet_int();
    __VERIFIER_assume((((StateRobotPosx == -10) && (StateRobotPosy == 6))));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume((StateRobotAct == UP || StateRobotAct == DOWN || StateRobotAct == LEFT || StateRobotAct == RIGHT || StateRobotAct == NONE));
}

int T = 0;
int main(void)
{
    initialize();
    compute_atomic_propositions();
    //@ assert (HitsWall == 0);
    //@ assert (HitsHole == 0);
    //@ assert ((T > 30) ==> (ReachesGoal == 1));
    while (1)
    {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        //@ assert (HitsWall == 0);
        //@ assert (HitsHole == 0);
        //@ assert ((T > 30) ==> (ReachesGoal == 1));
        printf("StateRobotPosx: %d, StateRobotPosy: %d, StateRobotAct: %d, HitsWall: %d, HitsHole: %d, ReachesGoal: %d\n", StateRobotPosx, StateRobotPosy, StateRobotAct, HitsWall, HitsHole, ReachesGoal);
        T++;
    }
}

