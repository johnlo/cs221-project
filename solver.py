import copy

def argmin(pairs):
    return None if len(pairs) == 0 else min(pairs, key=lambda p:p[1])[0]

# General code for representing a weighted CSP (Constraint Satisfaction Problem).
# All variables are being referenced by their index instead of their original
# names.
class CSP:
    def __init__(self):
        # Total number of variables in the CSP.
        self.numVars = 0

        # The list of variable names in the same order as they are added. A
        # variable name can be any hashable objects.
        self.varNames = []

        # Each entry is the list of domain values that its corresponding
        # variable can take on.
        # E.g. if B \in ['a', 'b'] is the second variable
        # then valNames[1] == ['a', 'b']
        self.valNames = []

        # Each entry is a unary potential table for the corresponding variable.
        # The potential table corresponds to the weight distribution of a variable
        # for all added unary potential functions. The table itself is a list
        # that has the same length as the variable's domain. If there's no
        # unary function, this table is stored as a None object.
        # E.g. if B \in ['a', 'b'] is the second variable, and we added two
        # unary potential functions f1, f2 for B,
        # then unaryPotentials[1][0] == f1('a') * f2('a')
        self.unaryPotentials = []

        # Each entry is a dictionary keyed by the index of the other variable
        # involved. The value is a binary potential table, where each table
        # stores the potential value for all possible combinations of
        # the domains of the two variables for all added binary potneital
        # functions. The table is represented as a 2D list, with size
        # dom(var) x dom(var2).
        #
        # As an example, if we only have two variables
        # A \in ['b', 'c'],  B \in ['a', 'b']
        # and we've added two binary functions f1(A,B) and f2(A,B) to the CSP,
        # then binaryPotentials[0][1][0][0] == f1('b','a') * f2('b','a').
        # binaryPotentials[0][0] should return a key error since a variable
        # shouldn't have a binary potential table with itself.
        #
        # One important thing to note here is that the indices in the potential
        # tables are indexed with respect to its variable's domain. Hence, 'b'
        # will have an index of 0 in A, but an index of 1 in B. Conversely, the
        # first value for A and B may not necessarily represent the same thing.
        # Beaware of the difference when implementing your CSP solver.
        self.binaryPotentials = []

    def add_variable(self, varName, domain):
        """
        Add a new variable to the CSP.
        """
        if varName in self.varNames:
            raise Exception("Variable name already exists: %s" % varName)
        var = len(self.varNames)
        self.numVars += 1
        self.varNames.append(varName)
        self.valNames.append(domain)
        self.unaryPotentials.append(None)
        self.binaryPotentials.append(dict())

    def add_unary_potential(self, varName, potentialFunc):
        """
        Add a unary potential function for a variable. Its potential
        value across the domain will be merged with any previously added
        unary potential functions through elementwise multiplication.
        """
        var = self.varNames.index(varName)
        potential = [float(potentialFunc(val)) for val in self.valNames[var]]
        if self.unaryPotentials[var] is not None:
            assert len(self.unaryPotentials[var]) == len(potential)
            self.unaryPotentials[var] = [self.unaryPotentials[var][i] * \
                potential[i] for i in range(len(potential))]
        else:
            self.unaryPotentials[var] = potential

    def add_binary_potential(self, varName1, varName2, potential_func):
        """
        Takes two variables |var1| and |var2| and a binary potential function
        |potentialFunc|, add to binaryPotentials. If |var1| and |var2| already
        had binaryPotentials added earlier, they will be merged through element
        wise multiplication.
        """
        var1 = self.varNames.index(varName1)
        var2 = self.varNames.index(varName2)
        self.update_binary_potential_table(var1, var2,
            [[float(potential_func(val1, val2)) \
                for val2 in self.valNames[var2]] for val1 in self.valNames[var1]])
        self.update_binary_potential_table(var2, var1, \
            [[float(potential_func(val1, val2)) \
                for val1 in self.valNames[var1]] for val2 in self.valNames[var2]])

    def update_binary_potential_table(self, var1, var2, table):
        """
        Update the binary potential table for binaryPotentials[var1][var2].
        If it exists, element-wise multiplications will be performed to merge
        them together.
        """
        if var2 not in self.binaryPotentials[var1]:
            self.binaryPotentials[var1][var2] = table
        else:
            currentTable = self.binaryPotentials[var1][var2]
            assert len(table) == len(currentTable)
            assert len(table[0]) == len(currentTable[0])
            for i in range(len(table)):
                for j in range(len(table[i])):
                    currentTable[i][j] *= table[i][j]

class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP sovler. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keey track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print "Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations)
            print "First assignment took %d operations" % self.firstAssignmentNumOperations
        else:
            print "No solution was found."

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param var: Index of an unassigned variable.
        @param val: Index of the proposed value with resepct to |var|'s domain.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert assignment[var] is None
        w = 1.0
        if self.csp.unaryPotentials[var]:
            w *= self.csp.unaryPotentials[var][val]
            if w == 0: return w
        for var2, potential in self.csp.binaryPotentials[var].iteritems():
            if assignment[var2] == None: continue  # Not assigned yet
            w *= potential[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, lcv = False, mac = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Monst Constrained Variable heuristics is used.
        @param lcv: When enabled, Least Constraining Value heuristics is used.
        @param mac: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.lcv = lcv
        self.mac = mac

        # Reset solutions from previous search.
        self.reset_results()

        # The list of domains of every variable in the CSP. Note that we only
        # use the indeces of the values. That is, if the domain of a variable
        # A is [2,3,5], then here, it will be stored as [0,1,2]. Original domain
        # name/value can be obtained from self.csp.valNames[A]
        self.domains = [list(range(len(domain))) for domain in self.csp.valNames]

        # Perform backtracking search.
        self.backtrack([None] * self.csp.numVars, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in range(self.csp.numVars):
                newAssignment[self.csp.varNames[var]] = self.csp.valNames[var][assignment[var]]
            self.allAssignments.append(newAssignment)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                # Map indices to real values for each variable
                self.optimalAssignment = newAssignment
                for var in range(self.csp.numVars):
                    self.optimalAssignment[self.csp.varNames[var]] = \
                        self.csp.valNames[var][assignment[var]]

                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the index of the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)

        # Obtain the order of which a variable's values will be tried. Note that
        # this stores the indices of the values with respect to |var|'s domain.
        ordered_values = self.get_ordered_values(assignment, var)

        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.mac:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None
        else:
            # Problem 1e
            # When arc consistency check is enabled.
            # BEGIN_YOUR_CODE (around 10 lines of code expected)
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    old_domains = copy.deepcopy(self.domains)
                    self.domains[var] = [val]
                    self.arc_consistency_check(var)
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    self.domains = old_domains
                    assignment[var] = None
            # END_YOUR_CODE

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return the index of a currently unassigned
        variable.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.

        @return var: Index of a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in xrange(len(assignment)):
                if assignment[var] is None: return var
        else:
            # Problem 1c
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # BEGIN_YOUR_CODE (around 7 lines of code expected)
            def numConsistentAssignments(var):
                return len([val for val in xrange(len(self.csp.valNames[var]))
                            if self.get_delta_weight(assignment, var, val) > 0])
            unassignedVars = [(var, numConsistentAssignments(var))
                              for var in xrange(len(assignment))
                              if assignment[var] is None]
            return argmin(unassignedVars)
            # END_YOUR_CODE

    def get_ordered_values(self, assignment, var):
        """
        Given an unassigned variable and a partial assignment, return an ordered
        list of indices of the variable's domain such that the backtracking
        algorithm will try |var|'s values according to this order.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.
        @param var: The variable that's going to be assigned next.

        @return ordered_values: A list of indeces of |var|'s domain values.
        """

        if not self.lcv:
            # Return an order of value indices without any heuristics.
            return self.domains[var]
        else:
            # Problem 1d
            # Heuristic: least constraining value (LCV)
            # Return value indices in ascending order of the number of additional
            # constraints imposed on unassigned neighboring variables.
            # BEGIN_YOUR_CODE (around 15 lines of code expected)
            def numConsistent(X_j, a, X_k):
                return len([b for b in self.domains[X_k]
                            if (self.get_delta_weight(assignment, X_k, b) > 0 and
                                self.csp.binaryPotentials[X_j][X_k][a][b] != 0)])
            def totalNumConsistent(X_j, a):
                X_k_list = [X_k for X_k in xrange(len(self.csp.valNames))
                            if (X_k != X_j and assignment[X_k] is None and
                                self.neighbors(X_j, X_k))]
                return sum([numConsistent(X_j, a, X_k) for X_k in X_k_list])
            return sorted([val for val in self.domains[var]],
                          key=lambda val: totalNumConsistent(var, val),
                          reverse=True)
            # END_YOUR_CODE

    def neighbors(self, var1, var2):
        return var2 in self.csp.binaryPotentials[var1]

    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.

        While not required, you can also choose to add return values in this
        function if there's a need.
        """
        # BEGIN_YOUR_CODE (around 17 lines of code expected)
        def isArcConsistent(X_i, x_i, X_j):
            for x_j in self.domains[X_j]:
                if self.csp.binaryPotentials[X_i][X_j][x_i][x_j] != 0:
                    return True
            return False
        # enforces arc consistency on X_i with respect to X_j
        # return True if Domain_i changed, False otherwise
        def enforceArcConsistency(X_i, X_j):
            Domain_i = copy.deepcopy(self.domains[X_i])
            changed = False
            for x_i in Domain_i:
                if not isArcConsistent(X_i, x_i, X_j):
                    self.domains[X_i].remove(x_i)
                    changed = True
            return changed
        stack = []
        stack.insert(0, var)
        while len(stack) > 0:
            X_changed = stack.pop()
            for X_neighbor in [X_n for X_n in xrange(len(self.csp.valNames))
                               if (X_n != X_changed and
                                   self.neighbors(X_changed, X_n))]:
                neighborChanged = enforceArcConsistency(X_neighbor, X_changed)
                if neighborChanged:
                    stack.insert(0, X_neighbor)
        # END_YOUR_CODE
