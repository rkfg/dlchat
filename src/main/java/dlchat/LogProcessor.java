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

public class LogProcessor {
    public static final String SPECIALS = "!\"#$;%^:?*()[]{}<>«»',.–—=+";
    private Set<String> dictSet = new HashSet<>();
    private Map<String, Integer> freq = new HashMap<>();
    private Map<String, Integer> dict = new HashMap<>();
    private boolean countFreq;
    private InputStream is;

    public LogProcessor(String filename, boolean countFreq) throws FileNotFoundException {
        this(new FileInputStream(filename), countFreq);
    }

    public LogProcessor(InputStream is, boolean countFreq) {
        this.is = is;
        this.countFreq = countFreq;
    }

    public void start() throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            String lastNick = "";
            String lastLine = "";
            while ((line = br.readLine()) != null) {
                String[] nickContent = line.toLowerCase().split("\\|", 2);
                if (nickContent.length > 1) {
                    if (nickContent[0].equals(lastNick)) {
                        if (!lastLine.isEmpty()) {
                            if (!SPECIALS.contains(lastLine.substring(lastLine.length() - 1))) {
                                lastLine += ",";
                            }
                            lastLine += " " + nickContent[1];
                        } else {
                            lastLine = nickContent[1];
                        }
                    } else {
                        if (lastLine.isEmpty()) {
                            lastLine = nickContent[1];
                        }
                        processLine(lastLine);
                        lastLine = nickContent[1];
                        lastNick = nickContent[0];
                    }
                }
            }
        }
    }

    protected void processLine(String lastLine) {
        doProcessLine(lastLine, dictSet, false);
    }

    protected void doProcessLine(String lastLine, Collection<String> resultCollection, boolean addSpecials) {
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
            Integer count = freq.get(word);
            if (count == null) {
                freq.put(word, 1);
            } else {
                freq.put(word, count + 1);
            }
        }
    }

    public Set<String> getDictSet() {
        return dictSet;
    }

    public Map<String, Integer> getFreq() {
        return freq;
    }

    public void setDict(Map<String, Integer> dict) {
        this.dict = dict;
    }

    protected boolean processWords(Collection<String> words, List<Integer> wordIdxs) {
        for (String word : words) {
            Integer wordIdx = dict.get(word);
            if (wordIdx != null) {
                wordIdxs.add(wordIdx);
            } else {
                wordIdxs.add(0);
            }
        }
        if (!wordIdxs.isEmpty()) {
            wordIdxs.add(1); // end of line
            return true;
        }
        return false;
    }

}
