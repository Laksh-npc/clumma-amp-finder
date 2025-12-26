import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, HelpCircle, Download, Trash2 } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { toast } from "@/hooks/use-toast";
import predictionBg from "@/assets/prediction-bg.jpg";

interface PredictionResult {
  id: string;
  sequence: string;
  probability: number;
  prediction: "AMP" | "Non-AMP";
  confidence: "High" | "Medium" | "Low";
}

const VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY";
const MAX_SEQUENCE_LENGTH = 50;

const EXAMPLE_SEQUENCES = `>Example_AMP_1
KKKKKKKKKKKKKKKKKKKK
>Example_AMP_2
GIGKFLHSAKKFGKAFVGEIMNS
>Example_Non-AMP_1
QQQQQQQQQ
>Example_Non-AMP_2
ACDEFGHIKLMNPQRSTVWY`;

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const Predict = () => {
  const [input, setInput] = useState("");
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const validateSequence = (seq: string): boolean => {
    return seq.split("").every((char) => VALID_AMINO_ACIDS.includes(char.toUpperCase()));
  };

  const parseInput = (text: string): { id: string; sequence: string }[] => {
    const lines = text.trim().split("\n");
    const sequences: { id: string; sequence: string }[] = [];
    let currentId = "";
    let currentSeq = "";

    lines.forEach((line) => {
      const trimmedLine = line.trim();
      if (trimmedLine.startsWith(">")) {
        if (currentSeq) {
          sequences.push({ id: currentId || `Sequence ${sequences.length + 1}`, sequence: currentSeq });
        }
        currentId = trimmedLine.substring(1).trim();
        currentSeq = "";
      } else if (trimmedLine) {
        currentSeq += trimmedLine.toUpperCase().replace(/\s/g, "");
      }
    });

    if (currentSeq) {
      sequences.push({ id: currentId || `Sequence ${sequences.length + 1}`, sequence: currentSeq });
    }

    if (sequences.length === 0) {
      lines.forEach((line, index) => {
        const seq = line.trim().toUpperCase().replace(/\s/g, "");
        if (seq) {
          sequences.push({ id: `Sequence ${index + 1}`, sequence: seq });
        }
      });
    }

    return sequences;
  };

  const getConfidenceLevel = (prob: number): "High" | "Medium" | "Low" => {
    if (prob > 0.8 || prob < 0.2) return "High";
    if (prob >= 0.3 && prob <= 0.7) return "Medium";
    return "Low";
  };

  const handlePredict = async () => {
    if (!input.trim()) {
      toast({
        title: "No input",
        description: "Please enter at least one amino acid sequence",
        variant: "destructive",
      });
      return;
    }

    const sequences = parseInput(input);

    if (sequences.length === 0) {
      toast({
        title: "Invalid input",
        description: "No valid sequences found",
        variant: "destructive",
      });
      return;
    }

    const invalidSeqs = sequences.filter((s) => !validateSequence(s.sequence));
    if (invalidSeqs.length > 0) {
      toast({
        title: "Invalid amino acids",
        description: `Found invalid characters. Only ${VALID_AMINO_ACIDS} are allowed`,
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const payload = {
        items: sequences.map((s) => ({
          id: s.id,
          sequence: s.sequence.slice(0, MAX_SEQUENCE_LENGTH),
        })),
      };

      const resp = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(`Server error (${resp.status})`);
      }

      const data = await resp.json();
      const predictions: PredictionResult[] = (data?.results ?? []).map((r: any) => ({
        id: r.id,
        sequence: r.sequence,
        probability: Number(r.probability),
        prediction: r.prediction === "AMP" ? "AMP" : "Non-AMP",
        confidence: (["High", "Medium", "Low"].includes(r.confidence)
          ? r.confidence
          : getConfidenceLevel(Number(r.probability))) as PredictionResult["confidence"],
      }));

      setResults(predictions);
      toast({
        title: "Prediction complete",
        description: `Analyzed ${predictions.length} sequence${predictions.length > 1 ? "s" : ""}`,
      });
    } catch (err: any) {
      toast({
        title: "Prediction failed",
        description: err?.message || "Unexpected error",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadExample = () => {
    setInput(EXAMPLE_SEQUENCES);
    toast({
      title: "Example loaded",
      description: "Sample sequences loaded into input area",
    });
  };

  const handleClear = () => {
    setInput("");
    setResults([]);
  };

  const handleDownloadCSV = () => {
    if (results.length === 0) return;

    const headers = ["Sequence ID", "Sequence", "AMP Probability", "Prediction", "Confidence Level"];
    const rows = results.map((r) => [
      r.id,
      r.sequence,
      r.probability.toFixed(3),
      r.prediction,
      r.confidence,
    ]);

    const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clumma_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Download started",
      description: "Results exported to CSV",
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section with Background */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-info/5 to-success/5 py-12">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-10"
          style={{ backgroundImage: `url(${predictionBg})` }}
        />
        <div className="relative container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground">
              AMP Prediction Tool
            </h1>
            <p className="text-lg text-muted-foreground">
              Enter your amino acid sequences below to predict antimicrobial activity
            </p>
          </div>
        </div>
      </section>

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Input Section */}
        <Card className="mb-8 shadow-medium border-primary/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  Sequence Input
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="text-xs">
                          Enter sequences in FASTA format (starting with &gt;header) or plain text (one
                          sequence per line). Only standard amino acids are accepted. Sequences longer than 50
                          residues will be truncated.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </CardTitle>
                <CardDescription>
                  Enter amino acid sequences in FASTA format or plain text
                </CardDescription>
              </div>
              <Button onClick={handleLoadExample} variant="outline" size="sm">
                Load Example
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={EXAMPLE_SEQUENCES}
                className="font-mono text-sm min-h-[200px] resize-y bg-secondary/30"
              />
              <div className="absolute bottom-2 right-2 text-xs text-muted-foreground bg-background/80 px-2 py-1 rounded">
                {input.length} characters
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={handlePredict} disabled={isLoading} className="flex-1">
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Predict AMPs"
                )}
              </Button>
              <Button onClick={handleClear} variant="outline" disabled={isLoading}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results Section */}
        {results.length > 0 && (
          <Card className="shadow-medium border-primary/20">
            <CardHeader className="bg-gradient-to-r from-primary/5 to-success/5">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Prediction Results</CardTitle>
                  <CardDescription>{results.length} sequences analyzed</CardDescription>
                </div>
                <Button onClick={handleDownloadCSV} variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Download CSV
                </Button>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="text-left p-3 font-semibold">Sequence ID</th>
                      <th className="text-left p-3 font-semibold">Sequence</th>
                      <th className="text-right p-3 font-semibold">Probability</th>
                      <th className="text-center p-3 font-semibold">Prediction</th>
                      <th className="text-center p-3 font-semibold">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={index} className="border-b hover:bg-muted/30 transition-colors">
                        <td className="p-3 font-medium">{result.id}</td>
                        <td className="p-3 font-mono text-xs">
                          {result.sequence.length > 30
                            ? result.sequence.substring(0, 30) + "..."
                            : result.sequence}
                        </td>
                        <td className="p-3 text-right font-mono">{result.probability.toFixed(3)}</td>
                        <td className="p-3 text-center">
                          <Badge
                            className={
                              result.prediction === "AMP"
                                ? "bg-success text-success-foreground"
                                : "bg-warning text-warning-foreground"
                            }
                          >
                            {result.prediction}
                          </Badge>
                        </td>
                        <td className="p-3 text-center">
                          <Badge
                            variant={result.confidence === "High" ? "default" : "outline"}
                            className={result.confidence === "Low" ? "text-muted-foreground" : ""}
                          >
                            {result.confidence}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 p-4 bg-info-muted rounded-lg text-xs text-muted-foreground">
                <p className="font-semibold text-foreground mb-1">Interpretation Guide:</p>
                <ul className="space-y-1">
                  <li>• <span className="font-medium">AMP Probability:</span> Higher values indicate stronger predicted antimicrobial activity</li>
                  <li>• <span className="font-medium">Prediction:</span> AMP if probability ≥ 0.5, Non-AMP otherwise</li>
                  <li>• <span className="font-medium">Confidence:</span> High (&gt;0.8 or &lt;0.2), Medium (0.3-0.7), Low (0.2-0.3 or 0.7-0.8)</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};

export default Predict;
