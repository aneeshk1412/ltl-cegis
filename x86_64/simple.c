int count=0;

int main() {
    while(1) {
        //@ assert (count < 10);
        if (count > 9) break;
        count++;
    }
    return count;
}

