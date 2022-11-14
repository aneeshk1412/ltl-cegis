
#include <stdio.h>
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int UP = 104;
int DOWN = 103;
int LEFT = 102;
int RIGHT = 101;
int NONE = 100;

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
    if ((px >= 1 && px <= 5) && (py >= 1 && py <= 6)) return 0;
    return 1;
}
int check_prop_GOAL(int px, int py)
{
    if ((px == 5) && (py == 1)) return 1;
    return 0;
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

int HitsWALL = 0;
int HitsHOLE = 0;
int ReachesGOAL = 0;

void compute_atomic_propositions(void)
{
    HitsWALL = check_prop_WALL(StateRobotPosx, StateRobotPosy) == 1;
    HitsHOLE = check_prop_HOLE(StateRobotPosx, StateRobotPosy) == 1;
    ReachesGOAL = check_prop_GOAL(StateRobotPosx, StateRobotPosy) == 1;
}

int policy(void)
{
    if (StateRobotAct == UP && 0) return UP;
    if (StateRobotAct == UP && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && (check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) && !check_prop_WALL(StateRobotPosx, StateRobotPosy - 1) && !check_prop_HOLE(StateRobotPosx, StateRobotPosy - 1)) return DOWN;
    if (StateRobotAct == UP && 0) return LEFT;
    if (StateRobotAct == UP && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && !check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) return RIGHT;
    if (StateRobotAct == UP && check_prop_GOAL(StateRobotPosx, StateRobotPosy)) return NONE;
    if (StateRobotAct == DOWN && 0) return UP;
    if (StateRobotAct == DOWN && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && (check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) && !check_prop_WALL(StateRobotPosx, StateRobotPosy - 1) && !check_prop_HOLE(StateRobotPosx, StateRobotPosy - 1)) return DOWN;
    if (StateRobotAct == DOWN && 0) return LEFT;
    if (StateRobotAct == DOWN && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && !check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) return RIGHT;
    if (StateRobotAct == DOWN && check_prop_GOAL(StateRobotPosx, StateRobotPosy)) return NONE;
    if (StateRobotAct == LEFT && 0) return UP;
    if (StateRobotAct == LEFT && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && (check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) && !check_prop_WALL(StateRobotPosx, StateRobotPosy - 1) && !check_prop_HOLE(StateRobotPosx, StateRobotPosy - 1)) return DOWN;
    if (StateRobotAct == LEFT && 0) return LEFT;
    if (StateRobotAct == LEFT && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && !check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) return RIGHT;
    if (StateRobotAct == LEFT && check_prop_GOAL(StateRobotPosx, StateRobotPosy)) return NONE;
    if (StateRobotAct == RIGHT && 0) return UP;
    if (StateRobotAct == RIGHT && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && (check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) || check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) && !check_prop_WALL(StateRobotPosx, StateRobotPosy - 1) && !check_prop_HOLE(StateRobotPosx, StateRobotPosy - 1)) return DOWN;
    if (StateRobotAct == RIGHT && 0) return LEFT;
    if (StateRobotAct == RIGHT && !check_prop_GOAL(StateRobotPosx, StateRobotPosy) && !check_prop_WALL(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 1, StateRobotPosy) && !check_prop_HOLE(StateRobotPosx + 2, StateRobotPosy)) return RIGHT;
    if (StateRobotAct == RIGHT && check_prop_GOAL(StateRobotPosx, StateRobotPosy)) return NONE;
    if (StateRobotAct == NONE && 0) return UP;
    if (StateRobotAct == NONE && 0) return DOWN;
    if (StateRobotAct == NONE && 0) return LEFT;
    if (StateRobotAct == NONE && 0) return RIGHT;
    if (StateRobotAct == NONE && 1) return NONE;
}

void initialize(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    StateRobotPosy = __VERIFIER_nondet_int();
    __VERIFIER_assume((((StateRobotPosx == 1) && (StateRobotPosy == 6))));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume((StateRobotAct == UP || StateRobotAct == DOWN || StateRobotAct == LEFT || StateRobotAct == RIGHT || StateRobotAct == NONE));
}

int T = 0;
int main(void)
{
    initialize();
    compute_atomic_propositions();
    //@ assert ((T > 5 + 6) ==> (ReachesGOAL == 1));
    //@ loop invariant (HitsWALL == 0);
    //@ loop invariant (HitsHOLE == 0);
    while (1)
    {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        //@ assert ((T > 5 + 6) ==> (ReachesGOAL == 1));
        printf("StateRobotPosx: %d, StateRobotPosy: %d, StateRobotAct: %d, HitsWALL: %d, HitsHOLE: %d, ReachesGOAL: %d\n", StateRobotPosx, StateRobotPosy, StateRobotAct, HitsWALL, HitsHOLE, ReachesGOAL);
        T++;
    }
}

