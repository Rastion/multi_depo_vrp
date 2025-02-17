from qubots.base_problem import BaseProblem
import math, random
import os

PENALTY = 1e9

class MDVRPProblem(BaseProblem):
    """
    Multi Depot Vehicle Routing Problem (MDVRP) for Qubots.

    In this problem, a fleet of trucks (with uniform load capacity and route duration capacity)
    is available at several depots. Each customer has a known demand and requires a certain service time.
    Trucks must serve customers such that:
      - Each customer is served exactly once.
      - The total demand on any truck does not exceed its capacity.
      - The total time spent on a route (travel distance plus service times) does not exceed the truck's route duration capacity.
    In addition, a depot is considered opened (and incurs a fixed cost) if at least one truck departs from it.
    
    **Candidate Solution Representation:**
      A dictionary with key "routes" mapping to a two-dimensional list of size [nb_depots][nb_trucks_per_depot],
      where each element is a list of customer indices (0-indexed) representing the sequence of customers served by that truck.
    """

    def __init__(self, instance_file: str, **kwargs):
        (self.nb_trucks_per_depot, self.nb_customers, self.nb_depots,
         self.route_duration_capacity, self.truck_capacity, self.demands,
         self.service_time, self.dist_matrix, self.dist_depots,
         self.opening_route_cost, self.opening_depots_cost) = self._read_instance(instance_file)

    def _read_instance(self, filename: str):
        """
        Reads an instance file in the Cordeau_2011 (MDVRP) format.

        Expected format:
          - First line: data type (ignored) and then three numbers:
              [ignored] nb_trucks_per_depot nb_customers nb_depots
          - Next, for each depot (nb_depots lines):
              Two numbers: route duration capacity and truck load capacity.
          - Then, for each customer (nb_customers lines):
              Customer id, X coordinate, Y coordinate, service time, demand.
          - Then, for each depot (nb_depots lines):
              Depot id, X coordinate, Y coordinate.
          - Finally, the instance uses additional data (if any) for other problems; only the above are used.

        This method also computes:
          - distance_matrix: a matrix (nb_customers x nb_customers) of Euclidean distances among customers.
          - dist_depots: a matrix (nb_depots x nb_customers) of Euclidean distances from each depot to each customer.
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            lines = f.readlines()
        nb_line = 0
        # First line: split tokens.
        datas = lines[nb_line].split()
        # In this format, assume tokens[1], tokens[2], tokens[3] correspond to:
        nb_trucks_per_depot = int(datas[1])
        nb_customers = int(datas[2])
        nb_depots = int(datas[3])
        
        # Read depot capacities (route duration capacity and truck capacity) for each depot.
        route_duration_capacity = [0] * nb_depots
        truck_capacity = [0] * nb_depots
        for d in range(nb_depots):
            nb_line += 1
            capacities = lines[nb_line].split()
            route_duration_capacity[d] = int(capacities[0])
            truck_capacity[d] = int(capacities[1])
        
        # Read customers data.
        nodes_xy = [[0, 0] for _ in range(nb_customers)]
        service_time = [0] * nb_customers
        demands = [0] * nb_customers
        for n in range(nb_customers):
            nb_line += 1
            customer = lines[nb_line].split()
            # customer[0] is id (ignored), customer[1] and customer[2] are coordinates.
            nodes_xy[n] = [float(customer[1]), float(customer[2])]
            service_time[n] = int(customer[3])
            demands[n] = int(customer[4])
        
        # Read depot coordinates.
        depot_xy = [[0, 0] for _ in range(nb_depots)]
        for d in range(nb_depots):
            nb_line += 1
            depot = lines[nb_line].split()
            depot_xy[d] = [float(depot[1]), float(depot[2])]
        
        # For this MDVRP instance, we also need two cost parameters.
        # Assume that opening_route_cost and opening_depots_cost are provided in the instance header.
        # (If not, we use default values; here we assume they are in the first line tokens 4 and 5.)
        if len(datas) >= 5:
            opening_route_cost = int(datas[4])
        else:
            opening_route_cost = 0
        # For depots, we assume a list of opening costs follows; if not, use a default for each depot.
        opening_depots_cost = [0] * nb_depots
        if len(datas) >= 5 + nb_depots:
            for d in range(nb_depots):
                opening_depots_cost[d] = float(datas[5 + d])
        else:
            # Use zero if not provided.
            opening_depots_cost = [0] * nb_depots
        
        # Compute distance matrix among customers.
        distance_matrix = self._compute_distance_matrix_customers(nodes_xy)
        # Compute distance matrix from each depot to customers.
        distance_depots = self._compute_distance_depots(depot_xy, nodes_xy)
        
        return (nb_trucks_per_depot, nb_customers, nb_depots, route_duration_capacity,
                truck_capacity, demands, service_time, distance_matrix, distance_depots,
                opening_route_cost, opening_depots_cost)

    def _compute_distance_matrix_customers(self, nodes_xy):
        nb = len(nodes_xy)
        matrix = [[0.0 for _ in range(nb)] for _ in range(nb)]
        for i in range(nb):
            for j in range(i, nb):
                d = math.sqrt((nodes_xy[i][0]-nodes_xy[j][0])**2 + (nodes_xy[i][1]-nodes_xy[j][1])**2)
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix

    def _compute_distance_depots(self, depot_xy, nodes_xy):
        nb_c = len(nodes_xy)
        nb_d = len(depot_xy)
        matrix = [[0.0 for _ in range(nb_c)] for _ in range(nb_d)]
        for d in range(nb_d):
            for i in range(nb_c):
                d_val = math.sqrt((depot_xy[d][0]-nodes_xy[i][0])**2 + (depot_xy[d][1]-nodes_xy[i][1])**2)
                matrix[d][i] = d_val
        return matrix

    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.

        Expects:
          solution: a dictionary with key "routes" mapping to a 2D list of size [nb_depots][nb_trucks_per_depot],
                    where each element is a list of customer indices (0-indexed) representing the route of that truck.
        Returns:
          The total cost, defined as the sum over all routes of (opening_route_cost + route distance)
          plus the sum of depot opening costs for depots that are used.
          If any capacity or duration constraint is violated, returns a high penalty.
        """
        # Verify solution structure.
        if not isinstance(solution, dict) or "routes" not in solution:
            return PENALTY
        routes = solution["routes"]
        if not isinstance(routes, list) or len(routes) != self.nb_depots:
            return PENALTY
        # Collect all customer indices from all routes.
        all_customers = []
        for d in range(self.nb_depots):
            if not isinstance(routes[d], list) or len(routes[d]) != self.nb_trucks_per_depot:
                return PENALTY
            for k in range(self.nb_trucks_per_depot):
                if not isinstance(routes[d][k], list):
                    return PENALTY
                all_customers.extend(routes[d][k])
        if sorted(all_customers) != list(range(self.nb_customers)):
            return PENALTY

        total_route_cost = 0.0
        depot_used = [False] * self.nb_depots

        # Evaluate each truck's route.
        for d in range(self.nb_depots):
            # Get the depot's distance array.
            depot_distances = self.dist_depots[d]
            for k in range(self.nb_trucks_per_depot):
                route = routes[d][k]
                if len(route) == 0:
                    continue
                # Mark depot as used.
                depot_used[d] = True
                # Total demand served by this route.
                route_quantity = sum(self.demands[i] for i in route)
                if route_quantity > self.truck_capacity[d]:
                    return PENALTY
                # Compute route service time.
                route_service_time = sum(self.service_time[i] for i in route)
                # Compute travel distance.
                route_distance = depot_distances[route[0]]  # from depot to first customer
                for i in range(len(route) - 1):
                    route_distance += self.dist_matrix[route[i]][route[i+1]]
                route_distance += depot_distances[route[-1]]  # from last customer back to depot
                # Check duration capacity if defined (if > 0).
                if self.route_duration_capacity[d] > 0:
                    if route_distance + route_service_time > self.route_duration_capacity[d]:
                        return PENALTY
                # Route cost: fixed opening cost plus travel distance.
                total_route_cost += self.opening_route_cost + route_distance

        depot_cost = 0.0
        for d in range(self.nb_depots):
            if depot_used[d]:
                depot_cost += self.opening_depots_cost[d]

        total_cost = total_route_cost + depot_cost
        return total_cost

    def random_solution(self):
        """
        Generates a random candidate solution.

        Randomly assigns each customer to one truck among all depots (total trucks = nb_depots * nb_trucks_per_depot)
        and then groups them by depot.
        """
        # Total number of trucks.
        total_trucks = self.nb_depots * self.nb_trucks_per_depot
        # Create a list for each truck.
        truck_routes = [[] for _ in range(total_trucks)]
        # Randomly assign each customer to a truck.
        for cust in range(self.nb_customers):
            r = random.randrange(total_trucks)
            truck_routes[r].append(cust)
        # Shuffle each truck's route.
        for r in range(total_trucks):
            random.shuffle(truck_routes[r])
        # Group routes by depot.
        routes = []
        for d in range(self.nb_depots):
            depot_routes = truck_routes[d*self.nb_trucks_per_depot:(d+1)*self.nb_trucks_per_depot]
            routes.append(depot_routes)
        return {"routes": routes}
