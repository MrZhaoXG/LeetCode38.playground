
//递归
func countAndSay(_ n: Int) -> String {
    if n == 1 {
        return "1"
    }
    let answer = countAndSay(n-1)
    var arr = [Int]()
    var index = 0
    var startIndex = 0
    while index != answer.count   {
        let indexOfStr = answer.index(answer.startIndex, offsetBy: index)
        if arr.last != Int(String(answer[indexOfStr])) {
            arr.append(index - startIndex)
            arr.append(Int(String(answer[indexOfStr]))!)
            startIndex = index
        }
        index += 1
    }
    arr.append(index - startIndex)
    arr.removeFirst()
    var result = ""
    for (index,number) in arr.enumerated() {
        if index % 2 == 1 {
            result += String(number) + String(arr[index - 1])
        }
    }
    return result
}
//回溯算法：不满足则回退一步
func backtrack(candidates: [Int], target: Int, result: inout [[Int]], index: Int, tmp: inout [Int]) {
    if target < 0 { return }
    if target == 0 {
        result.append(tmp)
        return
    }
    for start in index..<candidates.count {
        if target < candidates[start] { break }
        tmp.append(candidates[start])
        backtrack(candidates: candidates, target: target - candidates[start], result: &result, index: start + 1, tmp: &tmp)
        tmp.remove(at: tmp.count - 1)
    }
}

func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
    var result = [[Int]]()
    let candidates = candidates.sorted()
    var tmp = [Int]()
    backtrack(candidates: candidates, target: target, result: &result, index: 0, tmp: &tmp)
    var ans = [[Int]]()
    for i in result {
        if !ans.contains(i) {
            ans.append(i)
        }
    }
    return ans
}

//print(combinationSum([10,1,2,7,6,1,5], 8))

func firstMissingPositive(_ nums: [Int]) -> Int {
    var nums = nums.sorted()
    if nums.last ?? -1 < 0 { return 1 }
    nums = nums.filter { $0 > 0 }
    if nums.first ?? -1 != 1 { return 1}
    for index in 0...nums.count - 1 {
        if index > 0{
            if nums[index] == nums[index - 1] || nums[index] == nums[index - 1] + 1 {
                continue
            } else {
                return nums[index - 1] + 1
            }
        }
    }
    return (nums.last ?? 0) + 1
}

//print(firstMissingPositive([1,2,0]))

func trap(_ height: [Int]) -> Int {
    var ans = 0
    let size = height.count
    for i in 1..<size - 1 {
        var maxLeft = 0
        var maxRight = 0
        for j in stride(from: i, to: -1, by: -1) {
            maxLeft = max(height[j], maxLeft)
        }
        for j in i...size - 1 {
            maxRight = max(height[j], maxRight)
        }
        ans += min(maxRight, maxLeft) - height[i]
    }
    return ans
}
//print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))

func trap2(_ height: [Int]) -> Int {
    var ans = 0
    var leftMax = [Int]()
    var rightMax = [Int]()
    leftMax.insert(height[0], at: 0)
    for i in 1..<height.count {
        leftMax.append(max(height[i], leftMax.last ?? 0))
    }
    rightMax.insert(height.last ?? 0, at: 0)
    for j in stride(from: height.count - 2, to: -1, by: -1) {
        rightMax.insert(max(rightMax.first ?? 0, height[j]), at: 0)
    }
    for i in 1..<height.count - 1 {
        ans += min(leftMax[i], rightMax[i]) - height[i]
    }
    print(leftMax)
    print(rightMax)
    return ans
}

//print(trap2([0,1,0,2,1,0,1,3,2,1,2,1]))


//以下为字符串相乘（意义在于当字符串表示的数值已经大于Int64时候还能算出来）
    func carrierSolver(_ nums:inout [Int]) {
        var i = 0
        while i < nums.count {
            if nums[i] >= 10 {
                let carrier: Int = nums[i] / 10
                if i == nums.count - 1 {
                    nums.append(carrier)
                } else {
                    nums[i + 1] += carrier
                }
                nums[i] %= 10
            }
            i += 1
        }
    }

func stringMultiplyDigit(_ num1: String, _ num2: String) -> String {
    var tmp = [Int]()
    let num1 = num1.reversed()
    for (_, char) in num1.enumerated() {
        tmp.append(Int(String(char))! * Int(num2)!)
    }
    carrierSolver(&tmp)
    tmp = tmp.reversed()
    var res = ""
    for (_,d) in tmp.enumerated() {
        res += String(d)
    }
    return res
}

    
    func stringPlusString(_ str1: String, _ str2: String) -> String {
        var s1 = str1
        var s2 = str2
        if s1.count < s2.count {
            let tmp = s1
            s1 = s2
            s2 = tmp
        }
        var ints1 = [Int]()
        var ints2 = [Int]()
        for i in s1 {
            ints1.append(Int(String(i))!)
        }
        for i in s2 {
            ints2.append(Int(String(i))!)
        }
        ints1 = ints1.reversed()
        ints2 = ints2.reversed()
        for (index, int) in ints2.enumerated() {
            ints1[index] += int
        }
        carrierSolver(&ints1)
        ints1 = ints1.reversed()
        var ans = ""
        for (_, d) in ints1.enumerated() {
            ans += String(d)
        }
        return ans
    }

func multiply(_ num1: String, _ num2: String) -> String {
    if num1 == "0" || num2 == "0" {
        return "0"
    }
    let num2 = num2.reversed()
    var res = ""
    for (index, str) in num2.enumerated() {
        let tmpString = stringMultiplyDigit(num1
            , String(str)) + String(repeating: "0", count: index)
        res = stringPlusString(res, tmpString)
    }
    return res
}

//print(multiply("12323214321342134124122132131", "12332341213121324314123122321312"))//已经超出Int64 max 9223372036854775807

func isMatch(_ s: String, _ p: String) -> Bool {
    var patternIndex = 0
    var stringIndex = 0
    var match = 0
    var starIndex = -1
    
    let ss = s.utf8CString
    let sCount = ss.count - 1
    let pp = p.utf8CString
    let pCount = pp.count - 1
    
    let q = "?".utf8CString.first!
    let star = "*".utf8CString.first!
    
    while stringIndex < sCount {
        if patternIndex < pCount && (pp[patternIndex] == q || pp[patternIndex] == ss[stringIndex]) {
            stringIndex += 1
            patternIndex += 1
        } else if patternIndex < pCount && pp[patternIndex] == star {
            starIndex = patternIndex
            match = stringIndex
            patternIndex += 1
        } else if starIndex != -1 {
            patternIndex = starIndex + 1
            match += 1
            stringIndex = match
        } else {
            return false
        }
    }
    while patternIndex < pCount && pp[patternIndex] == star {
        patternIndex += 1
    }
    return patternIndex == pCount
}
//print(isMatch("acdcb", "a*cb"))

func jump(_ nums: [Int]) -> Int {
    if nums.count == 1 { return 0 }
    var step = 0
    var maxPosition = 0
    var nowPosition = 0
    while nowPosition < nums.count - 1 && nums[nowPosition] + nowPosition != nums.count - 1 {
        maxPosition = nowPosition + nums[nowPosition]
        var newMax = 0
        var oldMax = 0
        var indexOfMax = 0
        if maxPosition <= nums.count-1 {
            for i in nowPosition + 1...maxPosition {
                newMax = max(nums[i] + i, newMax)
                if newMax != oldMax {
                    oldMax = newMax
                    indexOfMax = i
                }
            }
            nowPosition = indexOfMax
            step += 1
        } else {
            step += 1
            break
        }
    }
    return nums[nowPosition] + nowPosition == nums.count - 1 ? step + 1 : step
}

//print(jump([2,3,1,1,4]))

func search(_ nums: inout [Int], res: inout [[Int]],  start: Int, n: Int) {
    if start == n {
        res.append(nums)
    }
    for i in start..<n {
        nums.swapAt(start, i)
        search(&nums, res: &res, start: start + 1, n: n)
        nums.swapAt(start, i)
    }
}

func permute(_ nums: [Int]) -> [[Int]] {
    var res = [[Int]]()
    var nums = nums
    search(&nums, res: &res, start: 0, n: nums.count)
    return res
}

//print(permute([1,2,3]))

func backtrack(res: inout [[Int]], nums: [Int], tmp: inout [Int], visited: inout [Int]) {
    if tmp.count == nums.count {
        res.append(tmp)
        return
    }
    for i in 0..<nums.count {
        if visited[i] == 1 { continue }
        visited[i] = 1;
        tmp.append(nums[i])
        backtrack(res: &res, nums: nums, tmp: &tmp, visited: &visited)
        visited[i] = 0
        tmp.removeLast()
    }
}

func permute2(_ nums: [Int]) -> [[Int]] {
    var res = [[Int]]()
    let nums = nums
    var visited: [Int] = Array(repeating: 0, count: nums.count)
    var tmp = [Int]()
    backtrack(res: &res, nums: nums, tmp: &tmp, visited: &visited)
    return res
}

//print(permute2([1,1,0,0,1,-1,-1,1]))

class PermuteUnique {
    func backtrack(res: inout [[Int]], nums: [Int], tmp: inout [Int], visited: inout [Int]) {
        if tmp.count == nums.count {
            res.append(tmp)
            return
        }
        for i in 0..<nums.count {
            if visited[i] == 0 {
                if i > 0 && nums[i] == nums[i - 1] && visited[i - 1] == 0 {
                    continue
                }
                visited[i] = 1
                tmp.append(nums[i])
                backtrack(res: &res, nums: nums, tmp: &tmp, visited: &visited)
                visited[i] = 0
                tmp.removeLast()
            }
        }
    }
    
    func permute2(_ nums: [Int]) -> [[Int]] {
        var res = [[Int]]()
        let nums = nums.sorted()
        var visited: [Int] = Array(repeating: 0, count: nums.count)
        var tmp = [Int]()
        backtrack(res: &res, nums: nums, tmp: &tmp, visited: &visited)
        return res
    }
}
let test = PermuteUnique()
//print(test.permute2([1,1,0,0,1,-1,-1,1]))

func rotate(_ matrix: inout [[Int]]) {
    let rows = matrix.count
    let columns = matrix[0].count
    for i in 0..<rows {
        for j in i + 1..<columns {
            let tmp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = tmp
        }
    }
    for i in 0..<rows {
        for j in 0..<columns / 2 {
            let tmp = matrix[i][j]
            matrix[i][j] = matrix[i][columns - 1 - j]
            matrix[i][columns - 1 - j] = tmp
        }
    }
}
var original = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]
//rotate(&original)
//print(original)

func groupAnagrams(_ strs: [String]) -> [[String]] {
    var dict2string = [Utf8dict : [String]]()
    typealias Utf8dict = [Int8 : Int]
    var str2utf8dict = [String: Utf8dict]()
    let strs = strs
    for str in strs {
        let utf8Ints = str.utf8CString.dropLast() as [Int8]
        var dict4SingleString = Utf8dict()
        for utf8Int in utf8Ints {
            if dict4SingleString[utf8Int] != nil {
                dict4SingleString[utf8Int]! += 1
            } else {
                dict4SingleString[utf8Int] = 1
            }
        }
        str2utf8dict[str] = dict4SingleString
        if dict2string[dict4SingleString] != nil {
            dict2string[dict4SingleString]!.append(str)
        } else {
            dict2string[dict4SingleString] = [str]
        }
    }
    return dict2string.map { $0.value }
}

//print(groupAnagrams(["abc","def","bac"]))
func multiply(x:  Double, n:  Int, res: inout Double, number: inout Int){
    if n == 1 {
        res *= x
        return
    }
    multiply(x: x * x, n: n / 2, res: &res, number: &number)
    number += 1
}


func myPow(_ x: Double, _ n: Int) -> Double {
    if n == 0 { return 1 }
    var positiveN = abs(n)
    var res: Double = 1
    var number = 0
    positiveN = positiveN % 2 == 0 ? positiveN : positiveN + 1
    multiply(x: x, n: positiveN, res: &res, number: &number)
    print(number)
    let remain: Int = positiveN - 2<<(number - 1)
    
    for _ in 0..<remain {
        res *= x
    }
    if positiveN != n { res /= x }
    if n > 0 { return res }
    else { return 1 / res }
}
//print(myPow(0.1,2147483646))

class SolutionForPower {
    func myPower(_ x: Double, _ n: Int) -> Double {
        var n = n
        var x = x
        if n < 0 {
            x = 1 / x
            n = -n
        }
        return fastPow(x, n)
    }
    private func fastPow(_ x: Double, _ n: Int) -> Double {
        if n == 0 {
            return 1
        }
        let half = fastPow(x, n / 2)
        if n % 2 == 0 {
            return half * half
        } else {
            return half * half * x
        }
    }
}

let power = SolutionForPower()
//print(power.myPower(0.1,2147483646))


class Solution {
    func solveNQueens(_ n: Int) -> [[String]] {
        guard n > 0 else {
            return []
        }
        var results = [[String]]()
        var cols = [Int]()
        cols.reserveCapacity(n)
        dfsHelper(n, &cols, &results)
        return results
    }
    
    fileprivate func dfsHelper(_ n: Int, _ cols: inout [Int], _ results: inout [[String]]) {
        if cols.count == n {
            results.append(draw(cols))
            return
        }
        
        for i in 0..<n {
            guard isValid(cols, i) else {
                continue
            }
            cols.append(i)
            dfsHelper(n, &cols, &results)
            cols.removeLast()
        }
    }
    
    
    fileprivate func isValid(_ cols: [Int], _ colIndex: Int) -> Bool {
        for rowIndex in 0..<cols.count {
            if colIndex == cols[rowIndex] {
                return false
            }
            if cols.count - rowIndex == colIndex - cols[rowIndex] {
                return false //"/"这个方向的row + col == const
            }
            if rowIndex - cols.count == colIndex - cols[rowIndex] {
                return false //"\"这个方向的row - col == const
            }
        }
        return true
    }
    
    fileprivate func draw(_ cols: [Int]) -> [String] {
        var result = [String]()
        for rowIndex in 0..<cols.count {
            var row = ""
            for j in 0..<cols.count {
                row += cols[rowIndex] == j ? "Q" : "."
            }
            result.append(row)
        }
        return result
    }
}

let queen = Solution()
//print(queen.solveNQueens(4))

func maxSubArray(_ nums: [Int]) -> Int {
    var ans = nums[0]
    var sum = 0
    for num in nums {
        if sum > 0 {
            sum += num
        } else {
            sum = num
        }
        ans = max(sum, ans)
    }
    return ans
}
//print(maxSubArray([1,2,3,1,2,3,1,2,1,-100,1,1,1,1,10,10]))

func spiralOrder(_ matrix: [[Int]]) -> [Int] {
    var res = [Int]()
    let columns = matrix[0].count
    let rows = matrix.count
    var added: [[Bool]] = Array(repeating: Array(repeating: false, count: columns), count: rows)
    var nextR = 0
    var nextC = 0
    var r = 0
    var c = 0
    var dr = [0,1,0,-1]
    var dc = [1,0,-1,0]
    var di = 0
    for _ in 0..<columns * rows {
        res.append(matrix[r][c])
        added[r][c] = true
        nextR = r + dr[di]
        nextC = c + dc[di]
        if nextR >= 0 && nextR < rows && nextC >= 0 && nextC < columns && !added[nextR][nextC] {
            r = nextR
            c = nextC
        } else {
            di = (di + 1) % 4
            r += dr[di]
            c += dc[di]
        }
    }
    return res
}

//print(spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
func backtrack(_ nums: [Int], nowPosition: inout Int, success: inout Bool, goodPosition: inout [Bool]) -> Bool {
    if nowPosition >= nums.count - 1 {
        success = true
        return true
    }
    if nums[nowPosition] == 0  {
        goodPosition[nowPosition] = false
        return false
    }
    let maxStep = nums[nowPosition]
    for index in stride(from: maxStep, to: 0, by: -1) {
        nowPosition += index
        if nowPosition < nums.count && goodPosition[nowPosition] {
            backtrack(nums, nowPosition: &nowPosition, success: &success, goodPosition: &goodPosition)
        }
        if success {
            break
        }
        nowPosition -= index
    }
    goodPosition[nowPosition] = false
    return success
}

func canJump(_ nums: [Int]) -> Bool {
    var goodPosition: [Bool] = Array(repeating: true, count: nums.count)
    var nowPosition = 0
    var successful = false
    backtrack(nums, nowPosition: &nowPosition, success: &successful, goodPosition: &goodPosition)
    return successful
}

//print(canJump([2,0,6,9,8,4,5,0,8,9,1,2,9,6,8,8,0,6,3,1,2,2,1,2,6,5,3,1,2,2,6,4,2,4,3,0,0,0,3,8,2,4,0,1,2,0,1,4,6,5,8,0,7,9,3,4,6,6,5,8,9,3,4,3,7,0,4,9,0,9,8,4,3,0,7,7,1,9,1,9,4,9,0,1,9,5,7,7,1,5,8,2,8,2,6,8,2,2,7,5,1,7,9,6]))

func insert(_ intervals:[[Int]], _ newInterval: [Int]) -> [[Int]] {
    var intervals = intervals
    intervals.append(newInterval)
    if intervals.count == 0 || intervals.count == 1 { return intervals }
    intervals = intervals.sorted(by: {(first, second) in
        if first[0] < second[0] {
            return true
        } else if first[0] == second[0] {
            return first[1] < second[1]
        } else {
            return false
        }
    })
    var ans: [[Int]] = [intervals[0]]
    for (index, interval) in intervals.enumerated() {
        if index == 0 { continue }
        let left = interval[0]
        let right = interval[1]
        let lastOneOfAnswerLeft = ans.last![0]
        let lastOneOfAnswerRight = ans.last![1]
        if left==lastOneOfAnswerLeft && right>=lastOneOfAnswerRight {
            ans[ans.count - 1] = interval
        } else if (lastOneOfAnswerRight>left && left>lastOneOfAnswerLeft && right>lastOneOfAnswerRight) || (left==lastOneOfAnswerRight && right>lastOneOfAnswerRight) {
            ans[ans.count - 1][1] = right
        } else if left > lastOneOfAnswerRight {
            ans.append(interval)
        }
    }
    return ans
}

//print(merge([[1,4],[4,5]]))
//print(insert([[1,3],[6,9]], [2,5]))
//print(insert([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]))

func lengthOfLastWord(_ s: String) -> Int {
    let uft8 = s.utf8CString.dropLast() as [Int8]
    if uft8 == Array(repeating: Int8(32), count: s.count) { return 0 }
    var ans = 0
    var s = s
    while s.last == " " {
        s = String(s.dropLast())
    }
    print(s)
    for index in stride(from: s.count - 1, to: -1, by: -1) {
        let indexOfString = s.index(s.startIndex, offsetBy: index)
        if s[indexOfString] == " " {
            break
        } else {
            ans += 1
        }
    }
    return ans
}

//print(lengthOfLastWord("a    hello     "))

func generateMatrix(_ n: Int) -> [[Int]] {
    var ans: [[Int]] = Array(repeating: Array(repeating: 0, count: n), count: n)
    var leftRow = 0
    var leftColumn = 0
    var rightRow = n - 1
    var rightColumn = n - 1
    var x = 1
    while leftRow <= rightRow {
        for i in leftColumn...rightColumn {
            ans[leftRow][i] = x
            x += 1
        }
        if leftRow + 1 <= rightRow - 1 {
            for i in leftRow + 1...rightRow - 1{
                ans[i][rightColumn] = x
                x += 1
            }
        }
        if rightColumn > leftColumn {
            for i in stride(from: rightColumn, to: leftColumn - 1, by: -1) {
                ans[rightRow][i] = x
                x += 1
            }
        }
        if rightRow - 1 >= leftRow + 1 {
            for i in stride(from: rightRow - 1, to: leftRow , by: -1) {
                ans[i][leftColumn] = x
                x += 1
            }
        }
        leftRow += 1
        leftColumn += 1
        rightColumn -= 1
        rightRow -= 1
    }
    return ans
}
//print(generateMatrix(3))


func getPermutation(_ n: Int, _ k: Int) -> String {
    var res = ""
    var n = n
    var indexOfTree = 0
    var newK = k
    var arr = [Int]()
    for i in 1...n {
        arr.append(i)
    }
    for _ in 1...n {
        var total = 1
        for i in 1...n {
            total *= i
        }
        indexOfTree = (newK - 1) / (total / n)
        let toBeRemoved = arr.remove(at: indexOfTree)
        res += String(toBeRemoved)
        newK -= (indexOfTree * (total / n))
        n -= 1
        print(toBeRemoved)
    }
    return res
}

//print(getPermutation(3, 3))

//61

public class ListNode {
    var val: Int
    var next: ListNode?
    init(_ val: Int) {
        self.val = val
    }
}

func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
    if head == nil || k == 0 { return head }
    var length = 0
    let head = head
    var nowNode: ListNode? = head
    var previousKNode: ListNode? = head
    var index = 0
    while nowNode != nil {
        length += 1
        nowNode = nowNode?.next
    }
    let k = k % length
    if k == 0 { return head }
    nowNode = head
    while nowNode?.next != nil {
        nowNode = nowNode?.next
        index += 1
        if index > k {
            previousKNode = previousKNode?.next
        }
    }
    let newHead = previousKNode?.next
    previousKNode?.next = nil
    nowNode?.next = head
    return newHead
}

var head = ListNode(0)
head.next = ListNode(1)
head.next?.next = ListNode(2)
//head.next?.next?.next = ListNode(3)
//head.next?.next?.next?.next = ListNode(4)
//var result = rotateRight(head, 3)
//while result != nil {
//    print(result?.val)
//    result = result?.next
//}

func uniquePaths(_ m: Int, _ n: Int) -> Int {
    if m == 1 || n == 1 { return 1 }
    var stepCount = m + n - 2
    var minDirection = min(m - 1, n - 1)
    let remain = stepCount - minDirection
    var steps = 1
    var mins = 1
    while stepCount != remain {
        steps *= stepCount
        stepCount -= 1
    }
    while minDirection != 0 {
        mins *= minDirection
        minDirection -= 1
    }
    return steps / mins
}

//print(uniquePaths(7, 3))

class SolutionForPath {
    func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
        var ans = 0
        var nowPosition = (0,0)
        backtrack(grid: obstacleGrid, ans: &ans, nowPosition: &nowPosition)
        return ans
    }
    
    func backtrack(grid: [[Int]], ans: inout Int, nowPosition: inout (Int, Int)) {
        if nowPosition == (grid.count - 1, grid[0].count - 1) && grid.count > 1{
            ans += 1
            return
        }
        if nowPosition.1 == grid[0].count - 1{
            for i in nowPosition.0...grid.count - 1 {
                if grid[i][nowPosition.1] == 1 {
                    return
                }
            }
            ans += 1
            return
        }
        if nowPosition.0 == grid.count - 1 {
            for i in nowPosition.1...grid[0].count - 1 {
                if grid[nowPosition.0][i] == 1 {
                    return
                }
            }
            ans += 1
            return
        }
        if grid[nowPosition.0][nowPosition.1] == 0 {
            nowPosition.1 += 1
            backtrack(grid: grid, ans: &ans, nowPosition: &nowPosition)
            nowPosition.1 -= 1
            nowPosition.0 += 1
            backtrack(grid: grid, ans: &ans, nowPosition: &nowPosition)
            nowPosition.0 -= 1
        }
    }
}

let solutionPath = SolutionForPath()
//print(solutionPath.uniquePathsWithObstacles([[0,0,0,1],[0,1,0,0]]))

//动态规划dynamic plan

func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
    if obstacleGrid[0][0] == 1 { return 0 }
    var ans = 0
    let rowCount = obstacleGrid.count
    let columnCount = obstacleGrid[0].count
    var grid = obstacleGrid
    if grid.count == 1 {
        if grid[0] == Array(repeating: 0, count: grid[0].count) {
            return 1
        } else {
            return 0
        }
    }
    if grid[0].count == 1 {
        if grid == Array(repeating: Array(repeating: 0, count: 1), count: grid.count) {
            return 1
        } else {
            return 0
        }
    }
    grid[0][0] = 1
    for i in 1...columnCount - 1 {
        if grid[0][i] == 0 {
            grid[0][i] = grid[0][i - 1]
        } else {
            grid[0][i] = 0
        }
    }
    for i in 1...rowCount - 1 {
        if grid[i][0] == 0 {
            grid[i][0] = grid[i - 1][0]
        } else {
            grid[i][0] = 0
        }
    }
    for i in 1...rowCount - 1 {
        for j in 1...columnCount - 1 {
            if grid[i][j] == 0 {
                grid[i][j] = grid[i][j - 1] + grid[i - 1][j]
            } else {
                grid[i][j] = 0
            }
        }
    }
    ans = grid[rowCount - 1][columnCount - 1]
    return ans
}

//print(uniquePathsWithObstacles([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,1],[0,0],[0,0],[1,0],[0,0],[0,0],[0,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,1],[0,0],[0,0],[0,0],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0]]))
func calculate(grid: [[Int]], i: Int, j: Int) -> Int {
    if i == grid.count || j == grid[0].count { return Int.max }
    if i == grid.count - 1 && j == grid[0].count - 1 { return grid[i][j] }
    return grid[i][j] + min(calculate(grid: grid, i: i + 1, j: j), calculate(grid: grid, i: i, j: j + 1))
}

func minPathSum(_ grid: [[Int]]) -> Int {
    return calculate(grid: grid, i: 0, j: 0)
}

func minPathSum2(_ grid: [[Int]]) -> Int {
    if grid.count == 1 {
        return grid[0].reduce(0, {$0 + $1})
    }
    if grid[0].count == 1 {
        return grid.map({$0[0]}).reduce(0, {$0+$1})
    }
    var grid = grid
    let rowCount = grid.count
    let columnCount = grid[0].count
    for i in 1...columnCount - 1 {
        grid[0][i] += grid[0][i - 1]
    }
    for i in 1...rowCount - 1 {
        grid[i][0] += grid[i - 1][0]
    }
    for i in 1...rowCount - 1 {
        for j in 1...columnCount - 1 {
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        }
    }
    return grid[rowCount - 1][columnCount - 1]
}

