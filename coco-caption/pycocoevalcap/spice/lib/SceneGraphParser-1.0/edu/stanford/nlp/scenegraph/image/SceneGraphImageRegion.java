package edu.stanford.nlp.scenegraph.image;

import java.util.List;
import java.util.Set;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphFactory;
import edu.stanford.nlp.semgraph.SemanticGraphFactory.Mode;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructure.Extras;
import edu.stanford.nlp.util.Generics;

public class SceneGraphImageRegion {

  public int h;
  public int w;
  public int x;
  public int y;

  public String phrase;

  public List<CoreLabel> tokens;

  public Set<SceneGraphImageAttribute> attributes = Generics.newHashSet();
  public Set<SceneGraphImageRelationship> relationships = Generics.newHashSet();

  public GrammaticalStructure gs;

  @SuppressWarnings("unchecked")
  public static SceneGraphImageRegion fromJSONObject(SceneGraphImage img, JSONObject obj) {

    SceneGraphImageRegion region = new SceneGraphImageRegion();
    region.h = ((Number) obj.get("h")).intValue();
    region.w = ((Number) obj.get("w")).intValue();
    region.x = ((Number) obj.get("x")).intValue();
    region.y = ((Number) obj.get("y")).intValue();

    region.phrase = (String) obj.get("phrase");


    if (obj.get("tokens") != null) {
      List<String> tokenStrings = (List<String>) obj.get("tokens");
      region.tokens = Generics.newArrayList(tokenStrings.size());
      for (String str : tokenStrings) {
        region.tokens.add(SceneGraphImageUtils.labelFromString(str));
      }
    }

    if (region.tokens != null && obj.get("gs") != null) {
      List<String> depTriplets = (List<String>) obj.get("gs");
      region.gs = SceneGraphImageUtils.getSemanticGraph(depTriplets, region.tokens);
    }

    return region;
  }





  @SuppressWarnings("unchecked")
  public JSONObject toJSONObject(SceneGraphImage sceneGraphImage) {
    JSONObject obj = new JSONObject();

    obj.put("h", this.h);
    obj.put("w", this.w);
    obj.put("x", this.x);
    obj.put("y", this.y);

    obj.put("phrase", this.phrase);

    if (this.tokens != null && ! this.tokens.isEmpty()) {
      JSONArray tokens = new JSONArray();
      for (CoreLabel lbl : this.tokens) {
        tokens.add(SceneGraphImageUtils.labelToString(lbl));
      }
      obj.put("tokens", tokens);
    }

    if (this.tokens != null && this.gs != null) {
      obj.put("gs", SceneGraphImageUtils.grammaticalStructureToJSON(this.gs));
    }

    return obj;
  }

  public SemanticGraph getBasicSemanticGraph() {
    return SemanticGraphFactory.makeFromTree(gs);
  }

  public SemanticGraph getEnhancedSemanticGraph() {
    return SemanticGraphFactory.makeFromTree(gs, Mode.CCPROCESSED, Extras.MAXIMAL, true, null);
  }

  public String toReadableString() {
    StringBuilder buf = new StringBuilder();
    buf.append(String.format("%-20s%-20s%-20s%n", "source", "reln", "target"));
    buf.append(String.format("%-20s%-20s%-20s%n", "---", "----", "---"));
    for (SceneGraphImageRelationship reln : this.relationships) {
      buf.append(String.format("%-20s%-20s%-20s%n", reln.subjectLemmaGloss(), reln.predicateLemmaGloss(), reln.objectLemmaGloss()));
    }

    for (SceneGraphImageAttribute attr: this.attributes) {
      buf.append(String.format("%-20s%-20s%-20s%n", attr.subjectLemmaGloss(), "is", attr.attributeLemmaGloss()));
    }
    return buf.toString();
  }

}

