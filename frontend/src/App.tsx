import React from 'react';
import { ChakraProvider, Box } from '@chakra-ui/react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';

// Pages
import Dashboard from './pages/Dashboard';
import ToxicityBatch from './pages/ToxicityBatch';
import BiasBatch from './pages/BiasBatch';
import RealtimeAnalysis from './pages/RealtimeAnalysis';
import Guardrails from './pages/Guardrails';
import AdversarialTesting from './pages/AdversarialTesting';
import Benchmarks from './pages/Benchmarks';
import ReliabilityTesting from './pages/ReliabilityTesting';
import PrivacyTesting from './pages/PrivacyTesting';
import HallucinationDetection from './pages/HallucinationDetection';

// Theme
import theme from './theme';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider theme={theme}>
        <Router>
          <Box minH="100vh" bg="gray.50">
            <Navbar />
            <Box display="flex">
              <Sidebar />
              <Box flex="1" p={8}>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  {/* Detection */}
                  <Route path="/toxicity-batch" element={<ToxicityBatch />} />
                  <Route path="/bias-batch" element={<BiasBatch />} />
                  <Route path="/realtime" element={<RealtimeAnalysis />} />
                  {/* Safety */}
                  <Route path="/guardrails" element={<Guardrails />} />
                  <Route path="/adversarial" element={<AdversarialTesting />} />
                  {/* Evaluation */}
                  <Route path="/benchmarks" element={<Benchmarks />} />
                  <Route path="/reliability" element={<ReliabilityTesting />} />
                  <Route path="/privacy" element={<PrivacyTesting />} />
                  <Route path="/hallucination" element={<HallucinationDetection />} />
                </Routes>
              </Box>
            </Box>
          </Box>
        </Router>
      </ChakraProvider>
    </QueryClientProvider>
  );
}

export default App;
