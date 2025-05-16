import torch
import torch.nn as nn
from torch import optim

class BoseHubbardNeuralNetwork(nn.Module):
    def __init__(self, num_sites, num_hidden=20):
        super(BoseHubbardNeuralNetwork, self).__init__()
        self.num_sites = num_sites
        self.num_hidden = num_hidden
        
        # Fully connected feedforward network
        self.input_layer = nn.Linear(num_sites, num_hidden)
        self.hidden_layer = nn.Linear(num_hidden, num_hidden)
        self.output_layer = nn.Linear(num_hidden, 2)  # 2 outputs: real and imaginary parts   
       
    def forward(self, x):
        # First layer with tanh activation
        hidden = torch.tanh(self.input_layer(x))
        # Output layer (real and imaginary parts of log psi)
        hidden_2 = torch.tanh(self.hidden_layer(hidden))

        output = self.output_layer(hidden_2)
        return output
    
    def psi(self, x):
        """Wave function ψ(n) = exp(out_1 + i*out_2)"""
        output = self(x)
        real = output[:, 0]
        imag = output[:, 1]
        return torch.exp(real + 1j * imag)

class BoseHubbardModel:
    def __init__(self, J, U, V, num_sites, num_particles,fock_states,nn:BoseHubbardNeuralNetwork):
        """
        Initialize the Bose-Hubbard model
        
        Args:
            J: Tunneling coefficient
            U: On-site interaction energy
            V: Site-dependent potential strength
            num_sites: Number of lattice sites
            num_particles: Total number of particles
            fock_states: Fock states
        """
        self.J = J
        self.U = U
        self.V = V
        self.num_sites = num_sites
        self.num_particles = num_particles
        self.fock_states = fock_states
        self.nn = nn

        # For 1D lattice
        if isinstance(num_sites, int):
            self.site_potential = V * ((torch.arange(num_sites) - 5) ** 2)

    def hamiltonian_expectation(self,state):
        """
        Calculate the sum of hamiltonian expectation product wave function give a sampled state.
        
        Args:
            state: Particle configuration (n₁, n₂, ...)

        Returns:
            sum of hamiltonian = Σ ⟨n|H|n'⟩ψ(n')
        """
        with torch.no_grad():
            result = torch.zeros(len(self.fock_states),dtype=torch.complex128)

            #Unicamente en el vector state esto tendra un valor distinto de 0.
            n_i_term =  torch.sum(self.V*state) + (self.U / 2)*torch.sum(state**2) - self.num_particles*(self.U / 2)
            
            #Genera un vector binario donde 0 si la fila es igual al estado, 1 si no.
            vector_binario = (self.fock_states == state).all(dim=1).float()        
            
            #Multiplicamos termino a termino el vector binario con el n_i_term, y lo sumamos al resultado.
            result += vector_binario*n_i_term

            #Termino de tunelaje
            for i in range(self.num_sites):
                j = (i+1)%self.num_sites
                
                #################################
                #Primer termino                 #
                #################################
                #Operacion de aniquilacion
                n_minus = self.fock_states[:,j].clone()
                fock_states_minus = self.fock_states.clone()
                mask_mayor_que_cero = fock_states_minus[:, j] > 0
                fock_states_minus[mask_mayor_que_cero, j] -= 1

                mask_igual_cero = mask_mayor_que_cero == False
                n_minus[mask_igual_cero] = 0
                fock_states_minus[mask_igual_cero] = 0
                            
                n_minus = (n_minus)**(1/2)

                #Operator de creacion
                n_plus = fock_states_minus[:,i].clone()
                fock_states_plus = fock_states_minus
                fock_states_plus[:,i] += 1

                n_plus = (n_plus+1)**(1/2)
                        
                #Genera un vector binario donde 0 si la fila es igual al estado, 1 si no
                vector_binario = (fock_states_plus == state).all(dim=1).float()
                result += -self.J*vector_binario*n_plus*n_minus



                #################################
                #Segundo termino                #
                #################################
                n_minus = self.fock_states[:,i].clone()
                fock_states_minus = self.fock_states.clone()
                mask_mayor_que_cero = fock_states_minus[:, i] > 0
                fock_states_minus[mask_mayor_que_cero, i] -= 1

                mask_igual_cero = mask_mayor_que_cero == False
                n_minus[mask_igual_cero] = 0
                fock_states_minus[mask_igual_cero] = 0
                            
                n_minus = (n_minus)**(1/2)

                #Operator de creacion
                n_plus = fock_states_minus[:,j].clone()
                fock_states_plus = fock_states_minus
                fock_states_plus[:,j] += 1

                n_plus = (n_plus+1)**(1/2)
                        
                #Genera un vector binario donde 0 si la fila es igual al estado, 1 si no
                vector_binario = (fock_states_plus == state).all(dim=1).float()
                result += -self.J*vector_binario*n_plus*n_minus

        wave_function = self.nn(self.fock_states)
        wave_function = torch.exp(wave_function[:,0] + 1j*wave_function[:,1])

        state_wave_function = self.nn(state)
        result = (result*wave_function)*torch.exp(state_wave_function[0] + 1j*state_wave_function[1]).conj()

        return torch.sum(result)
    
    def fit(self,epochs):

        optimizer = optim.Adam(self.nn.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
        
            probabilities = torch.sum(torch.abs(self.nn.psi(self.fock_states))**2).flatten()
            
            expectations = []

            for i in range(self.fock_states.shape[0]):
                hamiltonian_expectation = self.hamiltonian_expectation(self.fock_states[i])
                expectations.append(hamiltonian_expectation)

            expectations = torch.stack(expectations)/probabilities
            loss = expectations.sum().real
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
    
    def groud_state_energy(self):
        with torch.no_grad():
            probabilities = torch.sum(torch.abs(self.nn.psi(self.fock_states))**2).flatten()

            expectations = []
            for i in range(len(self.fock_states.shape[0])):
                hamiltonian_expectation = self.hamiltonian_expectation(self.fock_states[i])
                expectations.append(hamiltonian_expectation)
            
            expectations = torch.stack(expectations)/probabilities
            energy = expectations.sum().real

        return energy

