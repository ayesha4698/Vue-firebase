#include <stdio.h>

int main()
{
    char str[100];
    int i = 0, length = 0, flag = 0, start, end;

    printf("Input a string: ");
    // fgets(str, 100, stdin);
    fgets(str, 100, stdin);

    // Read in input from the command line
    while (i < 100 && str[i + 1] != '\0')
    {
        length = length + 1;
        i = i + 1;
    }
    strtok(str, "\n");

    // Find the length of the string.
    // Hint: How do you know when a string ends in C?

    // Check if str is a palindrome.

    end = length;
    start = 0;
    while (end > start)
    {
        if (str[end - 1] != str[start])
        {
            flag = 1;
            break;
        }
        end--;
        start++;
    }

    if (flag == 1)
        printf("%s is not a palindrome.\n", str);
    else
        printf("%s is a palindrome.\n", str);

    return 0;
}