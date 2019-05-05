# Reimplementation of Avg.absAvg() from rjust's lecture notes
#

# This is a weak implementation because a general 'Exception' is raised rather than
# a more specific exception type.

class Avg:

    @classmethod
    def absAvg(cls, numbers):
        if numbers is None or len(numbers) == 0:
            raise Exception("numbers must not be null or empty.")

        sum = 0
        for i in range(len(numbers)):
            d = numbers[i]
            if d < 0:
                sum -= d
            else:
                sum += d

        return sum/len(numbers)


