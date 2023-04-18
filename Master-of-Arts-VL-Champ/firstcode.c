#include <stdio.h>
#include <string.h>

int main(void)
{
  int wordsCount = 0;
  
  char wordsArray[100];
  fgets(wordsArray, sizeof(wordsArray), stdin);
  
  int len = strlen(wordsArray);
  int i;
  
  for (i = 0; i < len; i++)
  {
    if (wordsArray[i] != ' ')
    {
      if (i == 0 || wordsArray[i - 1] == ' ')
      {
        wordsCount++;
      }
    }
  }
  
  if (wordsArray[len - 1] == ' ' || wordsArray[len - 2] == ' ')
  {
    wordsCount--;
  }
  
  printf("%d\n", wordsCount);
  
  return 0;
}
