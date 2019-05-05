# Reimplementation of Avg.absAvg() from rjust's lecture notes
#

# Added EmptyInputException because of a weakness in original design which was highlighted by adding the
# raise suppression mutation.

class EmptyInputException(Exception):
    pass

class Avg:

    @classmethod
    def absAvg(cls, numbers):
        if numbers is None or len(numbers) == 0:
            raise EmptyInputException("numbers must not be null or empty.")

        sum = 0
        for i in range(len(numbers)):
            d = numbers[i]
            if d < 0:
                sum -= d
            else:
                sum += d

        return sum/len(numbers)


