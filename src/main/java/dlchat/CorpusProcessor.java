package dlchat;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CorpusProcessor {
    public static final String SPECIALS = "!\"#$;%^:?*()[]{}<>«»,.–—=+…";
    private Set<String> dictSet = new HashSet<>();
    private Map<String, Double> freq = new HashMap<>();
    private Map<String, Double> dict = new HashMap<>();
    private boolean countFreq;
    private InputStream is;
    private int rowSize;
    private String separator = " \\+\\+\\+\\$\\+\\+\\+ ";
    private int fieldsCount = 5;
    private int nameFieldIdx = 1;
    private int textFieldIdx = 4;

    public CorpusProcessor(String filename, int rowSize, boolean countFreq) throws FileNotFoundException {
        this(new FileInputStream(filename), rowSize, countFreq);
    }

    public CorpusProcessor(InputStream is, int rowSize, boolean countFreq) {
        this.is = is;
        this.rowSize = rowSize;
        this.countFreq = countFreq;
    }

    public void setFormatParams(String separator, int fieldsCount, int nameFieldIdx, int textFieldIdx){
        this.separator = separator;
        this.fieldsCount = fieldsCount;
        this.nameFieldIdx = nameFieldIdx;
        this.textFieldIdx = textFieldIdx;
    }
    
    public void start() throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            String lastName = "";
            String lastLine = "";
            while ((line = br.readLine()) != null) {
                String[] lineSplit = line.toLowerCase().split(separator, fieldsCount);
                if (lineSplit.length >= fieldsCount) {
                    // join consecuitive lines from the same speaker 
                    if (lineSplit[1].equals(lastName)) {
                        if (!lastLine.isEmpty()) {
                            // if the previous line doesn't end with a special symbol, append a comma and the current line
                            if (!SPECIALS.contains(lastLine.substring(lastLine.length() - 1))) {
                                lastLine += ",";
                            }
                            lastLine += " " + lineSplit[textFieldIdx];
                        } else {
                            lastLine = lineSplit[textFieldIdx];
                        }
                    } else {
                        if (lastLine.isEmpty()) {
                            lastLine = lineSplit[textFieldIdx];
                        } else {
                            processLine(lastLine);
                            lastLine = lineSplit[textFieldIdx];
                        }
                        lastName = lineSplit[nameFieldIdx];
                    }
                }
            }
            processLine(lastLine);
        }
    }

    protected void processLine(String lastLine) {
        tokenizeLine(lastLine, dictSet, false);
    }

    // here we not only split the words but also store punctuation marks
    protected void tokenizeLine(String lastLine, Collection<String> resultCollection, boolean addSpecials) {
        String[] words = lastLine.split("[ \t]");
        for (String word : words) {
            if (!word.isEmpty()) {
                boolean specialFound = true;
                while (specialFound && !word.isEmpty()) {
                    for (int i = 0; i < word.length(); ++i) {
                        int idx = SPECIALS.indexOf(word.charAt(i));
                        specialFound = false;
                        if (idx >= 0) {
                            String word1 = word.substring(0, i);
                            if (!word1.isEmpty()) {
                                addWord(resultCollection, word1);
                            }
                            if (addSpecials) {
                                addWord(resultCollection, String.valueOf(word.charAt(i)));
                            }
                            word = word.substring(i + 1);
                            specialFound = true;
                            break;
                        }
                    }
                }
                if (!word.isEmpty()) {
                    addWord(resultCollection, word);
                }
            }
        }
    }

    private void addWord(Collection<String> coll, String word) {
        if (coll != null) {
            coll.add(word);
        }
        if (countFreq) {
            Double count = freq.get(word);
            if (count == null) {
                freq.put(word, 1.0);
            } else {
                freq.put(word, count + 1);
            }
        }
    }

    public Set<String> getDictSet() {
        return dictSet;
    }

    public Map<String, Double> getFreq() {
        return freq;
    }

    public void setDict(Map<String, Double> dict) {
        this.dict = dict;
    }

    protected boolean wordsToIndexes(Collection<String> words, List<Double> wordIdxs) {
        int i = rowSize;
        for (String word : words) {
            if (--i == 0) {
                break;
            }
            Double wordIdx = dict.get(word);
            if (wordIdx != null) {
                wordIdxs.add(wordIdx);
            } else {
                wordIdxs.add(0.0);
            }
        }
        if (!wordIdxs.isEmpty()) {
            return true;
        }
        return false;
    }

}
