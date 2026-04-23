#include <iostream>
#include <string>

using namespace std;

struct Course {
    int id, s, e;
    Course* next;
};

bool hasConflict(Course* h, int s, int e, int ignore = -1) {
    for (Course* c = h; c; c = c->next) {
        if (c->id != ignore && s < c->e && e > c->s) return true;
    }
    return false;
}

void addCourse(Course*& h, int id, int s, int e) {
    Course** pp = &h;
    while (*pp && (*pp)->s < s) pp = &((*pp)->next);
    *pp = new Course{id, s, e, *pp};
}

void delCourse(Course*& h, int id) {
    Course **pp = &h;
    while (*pp && (*pp)->id != id) pp = &((*pp)->next);
    if (*pp) { 
        Course* tmp = *pp; 
        *pp = (*pp)->next; 
        delete tmp; 
    }
}

int main() {
    Course* head = nullptr;
    string cmd;
    while (cin >> cmd && cmd != "quit") {
        if (cmd == "insert") {
            int i, s, e; cin >> i >> s >> e;
            bool exists = false;
            for(Course* c = head; c; c = c->next) if(c->id == i) exists = true;
            if (exists || hasConflict(head, s, e)) cout << "false" << endl;
            else addCourse(head, i, s, e);
        } 
        else if (cmd == "delete") {
            int i; cin >> i; 
            delCourse(head, i);
        } 
        else if (cmd == "update") {
            int i, s, e; cin >> i >> s >> e;
            Course* target = nullptr;
            for(Course* c = head; c; c = c->next) if(c->id == i) target = c;
            if (!target || hasConflict(head, s, e, i)) cout << "false" << endl;
            else { 
                delCourse(head, i); 
                addCourse(head, i, s, e); 
            }
        } 
        else if (cmd == "print") {
            if (!head) cout << "Empty" << endl;
            else {
                for(Course* c = head; c; c = c->next) 
                    cout << c->id << ": " << c->s << " - " << c->e << endl;
            }
        }
    }
    while (head) { Course* t = head; head = head->next; delete t; }
    return 0;
}