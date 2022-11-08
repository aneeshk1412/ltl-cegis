//@ ltl invariant positive: []<>AP(reaches_checkpoint2 == 1) && []<>AP(reaches_checkpoint2 == 1);
extern void __VERIFIER_error() __attribute__ ((__noreturn__));
extern void __VERIFIER_assume() __attribute__ ((__noreturn__));
extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));

int LEFT = 101;
int RIGHT = 102;
int NONE = 100;

int reaches_checkpoint2 = 0;
int reaches_checkpoint1 = 0;
int pos = 0;
int act = 0;

void compute() {
    reaches_checkpoint2 = (pos == 3);
    reaches_checkpoint1 = (pos == -3);
}

void update() {
    if (act == 0) pos = pos - 1;
    if (act == 1) pos = pos + 1;
}

void getact() {
    if (pos == -4) act = 1;
    if (pos == 4) act = 0;
}

int main(void)
{
    pos = __VERIFIER_nondet_int();
    __VERIFIER_assume(pos > -5 && pos < 5);
    while (1) {
        getact();
        update();
        compute();
    }
}
