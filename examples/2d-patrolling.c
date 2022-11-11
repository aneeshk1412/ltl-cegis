//@ ltl invariant positive: []<>(AP(reaches_checkpoint1 == 1)) && []<>(AP(reaches_checkpoint2 == 1));

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
    if ((px >= 1 && px <= 10) && (py >= 1 && py <= 5)) return 0;
    return 1;
}
int check_prop_CHECKPOINT1(int px, int py)
{
    if ((px == 1) && (py == 1)) return 1;
    return 0;
}
int check_prop_CHECKPOINT2(int px, int py)
{
    if ((px == 10) && (py == 5)) return 1;
    return 0;
}

int HitsWALL = 0;
int ReachesCHECKPOINT1 = 0;
int ReachesCHECKPOINT2 = 0;

void compute_atomic_propositions(void)
{
    HitsWALL = check_prop_WALL(StateRobotPosx, StateRobotPosy) == 1;
    ReachesCHECKPOINT1 = check_prop_CHECKPOINT1(StateRobotPosx, StateRobotPosy) == 1;
    ReachesCHECKPOINT2 = check_prop_CHECKPOINT2(StateRobotPosx, StateRobotPosy) == 1;
}

int policy(void)
{
    INSERT_ASP
    return NONE;
}

void initialize(void)
{
    StateRobotPosx = __VERIFIER_nondet_int();
    StateRobotPosy = __VERIFIER_nondet_int();
    __VERIFIER_assume((((StateRobotPosx >= 1 && StateRobotPosx <= 10) && (StateRobotPosy >= 1 && StateRobotPosy <= 5))));
    StateRobotAct = __VERIFIER_nondet_int();
    __VERIFIER_assume((StateRobotAct == UP || StateRobotAct == DOWN || StateRobotAct == LEFT || StateRobotAct == RIGHT || StateRobotAct == NONE));
}

int T = 0;
int main(void)
{
    initialize();
    compute_atomic_propositions();
    //@ loop invariant (HitsWALL == 0);
    while (1)
    {
        StateRobotAct = policy();
        apply_StateRobotAct();
        compute_atomic_propositions();
        printf("StateRobotPosx: %d, StateRobotPosy: %d, StateRobotAct: %d, HitsWALL: %d, ReachesCHECKPOINT1: %d, ReachesCHECKPOINT2: %d\n", StateRobotPosx, StateRobotPosy, StateRobotAct, HitsWALL, ReachesCHECKPOINT1, ReachesCHECKPOINT2);
        T++;
    }
}

