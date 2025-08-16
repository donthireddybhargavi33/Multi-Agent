import re
import json
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    EMOTION_ANALYSIS = "Emotion Analysis"
    CODING = "Coding"
    PHYSICS_CHEMISTRY = "Physics/Chemistry"
    MATH = "Math"
    GENERAL_CONVERSATION = "General Conversation"
    AMBIGUOUS = "Ambiguous"

@dataclass
class DetectionResult:
    agent_type: AgentType
    confidence: float
    reasons: List[str]
    alternative_agents: List[Tuple[AgentType, float]]
    context_clues: List[str]

class EnhancedUnifiedAgentDetector:
    """Comprehensive agent detection with context-aware classification and edge case handling"""
    
    def __init__(self):
        # Comprehensive pattern database
        self.patterns = {
            AgentType.EMOTION_ANALYSIS: {
                'explicit': [
                    # Primary emotion expressions with synonyms and antonyms
                    r'\b(i am|i\'m|i feel|i\'m feeling|i felt|i was|i\'ve been|i\'m experiencing|i\'m going through|i\'m dealing with|i\'m struggling with|i\'m battling|i\'m facing|i\'m overwhelmed by|i\'m consumed by|i\'m plagued by|i\'m tormented by|i\'m haunted by)\s+(?:very|really|quite|so|extremely|slightly|kind of|sort of|utterly|completely|totally|absolutely|somewhat|rather|fairly|pretty|incredibly|unbelievably|painfully|acutely|intensely|mildly|subtly|vaguely|slightly|marginally|barely|hardly|scarcely)?\s*(?:sad|depressed|melancholy|sorrowful|down|blue|low|unhappy|miserable|dejected|despairing|despondent|disheartened|discouraged|crestfallen|heartbroken|grief-stricken|mournful|woeful|wretched|forlorn|gloomy|morose|somber|glum|sullen|moody|downcast|downhearted|dispirited|demoralized|crushed|devastated|shattered|broken|angry|furious|livid|irate|enraged|infuriated|incensed|wrathful|seething|fuming|irritated|annoyed|vexed|exasperated|aggravated|provoked|agitated|upset|disturbed|perturbed|disquieted|riled|irked|peeved|miffed|piqued|rankled|nettled|galled|antagonized|hostile|antagonistic|resentful|bitter|spiteful|malicious|vindictive|anxious|worried|nervous|apprehensive|fearful|afraid|scared|terrified|panicked|alarmed|uneasy|tense|stressed|distressed|agitated|restless|jittery|on edge|keyed up|worked up|fretful|edgy|jumpy|skittish|spooked|unnerved|daunted|intimidated|overwhelmed|daunted|coward|timid|shy|bashful|reticent|hesitant|reluctant|averse|loath|unwilling|disinclined|tired|exhausted|weary|fatigued|drained|spent|worn out|burned out|depleted|prostrate|enervated|debilitated|lethargic|sluggish|listless|apathetic|indifferent|uninterested|unmotivated|uninspired|dispirited|demoralized|disheartened|crestfallen|deflated|flat|empty|hollow|numb|vacant|blank|devoid|barren|desolate|bereft|bereaved|deprived|robbed|stripped|denuded|frustrated|exasperated|irritated|annoyed|vexed|provoked|aggravated|thwarted|balked|foiled|stymied|stumped|baffled|bewildered|perplexed|puzzled|confounded|confused|mystified|flummoxed|discombobulated|befuddled|addled|muddled|foggy|hazy|unclear|muddled|jumbled|tangled|knotted|snarled|confused|disoriented|lost|adrift|at sea|befuddled|befogged|clouded|muddied|murky|obscure|uncertain|unsure|doubtful|dubious|skeptical|suspicious|wary|cautious|chary|leery|hesitant|reluctant|averse|loath|unwilling|disinclined|lonely|alone|isolated|abandoned|forsaken|deserted|neglected|rejected|shunned|spurned|ostracized|excluded|shut out|left out|cast out|cast aside|forgotten|ignored|overlooked|disregarded|unnoticed|unseen|invisible|transparent|ghostly|phantom|spectral|empty|hollow|vacant|void|barren|desolate|bereft|bereaved|deprived|robbed|stripped|denuded|overwhelmed|overpowered|overcome|conquered|vanquished|subdued|subjugated|crushed|broken|shattered|devastated|destroyed|demolished|ruined|wrecked|undone|unmade|undone|ruined|wrecked|shattered|broken|disappointed|disillusioned|disenchanted|let down|crestfallen|chagrined|disgruntled|dissatisfied|unfulfilled|unmet|thwarted|balked|foiled|stymied|stumped|hopeful|optimistic|positive|confident|assured|certain|convinced|secure|sanguine|buoyant|cheerful|upbeat|encouraged|heartened|uplifted|inspired|motivated|energized|invigorated|animated|vitalized|revitalized|rejuvenated|renewed|refreshed|restored|recharged|reinvigorated|grateful|thankful|appreciative|obliged|indebted|beholden|mindful|conscious|aware|cognizant|sensible|sensitive|alive|awake|alert|attentive|receptive|responsive|open|receptive|available|accessible|approachable|reachable|attainable|achievable|feasible|viable|practicable|workable|content|satisfied|pleased|gratified|fulfilled|satiated|quenched|slaked|assuaged|alleviated|mitigated|ameliorated|improved|better|superior|enhanced|augmented|increased|boosted|elevated|raised|lifted|uplifted|buoyed|supported|sustained|maintained|preserved|protected|safeguarded|secured|shielded|defended|guarded|watched|monitored|supervised|overseen|managed|controlled|regulated|governed|directed|guided|led|conducted|steered|navigated|piloted|helmed|commanded|ruled|dominated|mastered|conquered|vanquished|overcome|surmounted|surpassed|exceeded|transcended|surpassed|excelled|shone|sparkled|dazzled|gleamed|glowed|radiated|beamed|shimmered|shone|sparkled|dazzled|gleamed|glowed|radiated|beamed|shimmered)\b',
                    
                    # Secondary emotion expressions
                    r'\b(makes? me|made me|leaves? me|leaving me|gets? me|getting me|leaves? me feeling|left me feeling|has me feeling|had me feeling|keeps? me|keeping me|renders? me|rendering me|reduces? me|reducing me|drives? me|driving me|pushes? me|pushing me|pulls? me|pulling me|brings? me|bringing me|carries? me|carrying me|takes? me|taking me|puts? me|putting me|sets? me|setting me|throws? me|throwing me|sends? me|sending me|puts? me in|puts? me into|throws? me into|sends? me into|plunges? me|plunging me|sinks? me|sinking me|drops? me|dropping me|lifts? me|lifting me|raises? me|raising me|elevates? me|elevating me|boosts? me|boosting me|cheers? me|cheering me|comforts? me|comforting me|soothes? me|soothing me|calms? me|calming me|relaxes? me|relaxing me|eases? me|easing me|relieves? me|relieving me|frees? me|freeing me|liberates? me|liberating me|releases? me|releasing me|unburdens? me|unburdening me|lightens? me|lightening me|heals? me|healing me|restores? me|restoring me|renews? me|renewing me|revives? me|reviving me|rejuvenates? me|rejuvenating me|refreshes? me|refreshing me|recharges? me|recharging me|reinvigorates? me|reinvigorating me|energizes? me|energizing me|vitalizes? me|vitalizing me|animates? me|animating me|inspires? me|inspiring me|motivates? me|motivating me|encourages? me|encouraging me|heartens? me|heartening me|uplifts? me|uplifting me|elevates? me|elevating me|exalts? me|exalting me|ennobles? me|ennobling me|dignifies? me|dignifying me|honors? me|honoring me|glorifies? me|glorifying me|magnifies? me|magnifying me|amplifies? me|amplifying me|augments? me|augmenting me|enhances? me|enhancing me|intensifies? me|intensifying me|heightens? me|heightening me|sharpens? me|sharpening me|focuses? me|focusing me|concentrates? me|concentrating me|centers? me|centering me|grounds? me|grounding me|anchors? me|anchoring me|stabilizes? me|stabilizing me|balances? me|balancing me|harmonizes? me|harmonizing me|aligns? me|aligning me|adjusts? me|adjusting me|adapts? me|adapting me|accommodates? me|accommodating me|conforms? me|conforming me|reconciles? me|reconciling me|integrates? me|integrating me|incorporates? me|incorporating me|assimilates? me|assimilating me|absorbs? me|absorbing me|engulfs? me|engulfing me|consumes? me|consuming me|devours? me|devouring me|overwhelms? me|overwhelming me|overpowers? me|overpowering me|overcomes? me|overcoming me|subdues? me|subduing me|subjugates? me|subjugating me|crushes? me|crushing me|breaks? me|breaking me|shatters? me|shattering me|destroys? me|destroying me|demolishes? me|demolishing me|ruins? me|ruining me|wrecks? me|wrecking me|undoes? me|undoing me|unmakes? me|unmaking me|erases? me|erasing me|obliterates? me|obliterating me|annihilates? me|annihilating me|extinguishes? me|extinguishing me|quenches? me|quenching me|suppresses? me|suppressing me|represses? me|repressing me|oppresses? me|oppressing me|depresses? me|depressing me|compresses? me|compressing me|condenses? me|condensing me|contracts? me|contracting me|shrinks? me|shrinking me|reduces? me|reducing me|diminishes? me|diminishing me|lessens? me|lessening me|minimizes? me|minimizing me|belittles? me|belittling me|diminishes? me|diminishing me|devalues? me|devaluing me|degrades? me|degrading me|debases? me|debasing me|corrupts? me|corrupting me|perverts? me|perverting me|distorts? me|distorting me|twists? me|twisting me|warps? me|warping me|bends? me|bending me|flexes? me|flexing me|stretches? me|stretching me|strains? me|straining me|stresses? me|stressing me|tensions? me|tensioning me|tightens? me|tightening me|constricts? me|constricting me|restricts? me|restricting me|limits? me|limiting me|confines? me|confining me|imprisons? me|imprisoning me|enslaves? me|enslaving me|binds? me|binding me|chains? me|chaining me|fetters? me|fettering me|shackles? me|shackling me|handcuffs? me|handcuffing me|manacles? me|manacling me|restrains? me|restraining me|constrains? me|constraining me|curtails? me|curtailing me|inhibits? me|inhibiting me|hinders? me|hindering me|impedes? me|impeding me|obstructs? me|obstructing me|blocks? me|blocking me|bars? me|barring me|prevents? me|preventing me|stops? me|stopping me|halts? me|halting me|arrests? me|arresting me|checks? me|checking me|curbs? me|curbing me|reins? me|reining me|bridles? me|bridling me|controls? me|controlling me|manages? me|managing me|directs? me|directing me|guides? me|guiding me|leads? me|leading me|steers? me|steering me|navigates? me|navigating me|pilots? me|piloting me|commands? me|commanding me|rules? me|ruling me|dominates? me|dominating me|masters? me|mastering me|conquers? me|conquering me|vanquishes? me|vanquishing me|overcomes? me|overcoming me|surmounts? me|surmounting me|surpasses? me|surpassing me|exceeds? me|exceeding me|transcends? me|transcending me|ascends? me|ascending me|soars? me|soaring me|flies? me|flying me|rises? me|rising me|elevates? me|elevating me|lifts? me|lifting me|raises? me|raising me|boosts? me|boosting me|propels? me|propelling me|drives? me|driving me|pushes? me|pushing me|pulls? me|pulling me|draws? me|drawing me|attracts? me|attracting me|allures? me|alluring me|entices? me|enticing me|lures? me|luring me|seduces? me|seducing me|charms? me|charming me|enchanting me|bewitches? me|bewitching me|captivates? me|captivating me|fascinates? me|fascinating me|intrigues? me|intriguing me|interests? me|interesting me|engages? me|engaging me|involves? me|involving me|absorbs? me|absorbing me|engrosses? me|engrossing me|immerses? me|immersing me|envelops? me|enveloping me|enfolds? me|enfolding me|embraces? me|embracing me|holds? me|holding me|cradles? me|cradling me|supports? me|supporting me|sustains? me|sustaining me|maintains? me|maintaining me|preserves? me|preserving me|protects? me|protecting me|safeguards? me|safeguarding me|secures? me|securing me|shields? me|shielding me|defends? me|defending me|guards? me|guarding me|watches? me|watching me|monitors? me|monitoring me|supervises? me|supervising me|oversees? me|overseeing me|manages? me|managing me|controls? me|controlling me|regulates? me|regulating me|governs? me|governing me|directs? me|directing me|guides? me|guiding me|leads? me|leading me|steers? me|steering me|navigates? me|navigating me|pilots? me|piloting me|commands? me|commanding me|rules? me|ruling me|dominates? me|dominating me|masters? me|mastering me)\s+(?:feel|feeling)?\s*(?:sad|depressed|melancholy|sorrowful|down|blue|low|unhappy|miserable|dejected|despairing|despondent|disheartened|discouraged|crestfallen|heartbroken|grief-stricken|mournful|woeful|wretched|forlorn|gloomy|morose|somber|glum|sullen|moody|downcast|downhearted|dispirited|demoralized|crushed|devastated|shattered|broken|angry|furious|livid|irate|enraged|infuriated|incensed|wrathful|seething|fuming|irritated|annoyed|vexed|exasperated|aggravated|provoked|agitated|upset|disturbed|perturbed|disquieted|riled|irked|peeved|miffed|piqued|rankled|nettled|galled|antagonized|hostile|antagonistic|resentful|bitter|spiteful|malicious|vindictive|anxious|worried|nervous|apprehensive|fearful|afraid|scared|terrified|panicked|alarmed|uneasy|tense|stressed|distressed|agitated|restless|jittery|on edge|keyed up|worked up|fretful|edgy|jumpy|skittish|spooked|unnerved|daunted|intimidated|overwhelmed|daunted|coward|timid|shy|bashful|reticent|hesitant|reluctant|averse|loath|unwilling|disinclined|tired|exhausted|weary|fatigued|drained|spent|worn out|burned out|depleted|prostrate|enervated|debilitated|lethargic|sluggish|listless|apathetic|indifferent|uninterested|unmotivated|uninspired|dispirited|demoralized|disheartened|crestfallen|deflated|flat|empty|hollow|numb|vacant|blank|devoid|barren|desolate|bereft|bereaved|deprived|robbed|stripped|denuded|frustrated|exasperated|irritated|annoyed|vexed|provoked|aggravated|thwarted|balked|foiled|stymied|stumped|baffled|bewildered|perplexed|puzzled|confounded|confused|mystified|flummoxed|discombobulated|befuddled|addled|muddled|foggy|hazy|unclear|muddled|jumbled|tangled|knotted|snarled|confused|disoriented|lost|adrift|at sea|befuddled|befogged|clouded|muddied|murky|obscure|uncertain|unsure|doubtful|dubious|skeptical|suspicious|wary|cautious|chary|leery|hesitant|reluctant|averse|loath|unwilling|disinclined|lonely|alone|isolated|abandoned|forsaken|deserted|neglected|rejected|shunned|spurned|ostracized|excluded|shut out|left out|cast out|cast aside|forgotten|ignored|overlooked|disregarded|unnoticed|unseen|invisible|transparent|ghostly|phantom|spectral|empty|hollow|vacant|void|barren|desolate|bereft|bereaved|deprived|robbed|stripped|denuded|overwhelmed|overpowered|overcome|conquered|vanquished|subdued|subjugated|crushed|broken|shattered|devastated|destroyed|demolished|ruined|wrecked|undone|unmade|undone|ruined|wrecked|shattered|broken|disappointed|disillusioned|disenchanted|let down|crestfallen|chagrined|disgruntled|dissatisfied|unfulfilled|unmet|thwarted|balked|foiled|stymied|stumped|hopeful|optimistic|positive|confident|assured|certain|convinced|secure|sanguine|buoyant|cheerful|upbeat|encouraged|heartened|uplifted|inspired|motivated|energized|invigorated|animated|vitalized|revitalized|rejuvenated|renewed|refreshed|restored|recharged|reinvigorated|grateful|thankful|appreciative|obliged|indebted|beholden|mindful|conscious|aware|cognizant|sensible|sensitive|alive|awake|alert|attentive|receptive|responsive|open|receptive|available|accessible|approachable|reachable|attainable|achievable|feasible|viable|practicable|workable|content|satisfied|pleased|gratified|fulfilled|satiated|quenched|slaked|assuaged|alleviated|mitigated|ameliorated|improved|better|superior|enhanced|augmented|increased|boosted|elevated|raised|lifted|uplifted|buoyed|supported|sustained|maintained|preserved|protected|safeguarded|secured|shielded|defended|guarded|watched|monitored|supervised|overseen|managed|controlled|regulated|governed|directed|guided|led|conducted|steered|navigated|piloted|helmed|commanded|ruled|dominated|mastered|conquered|vanquished|overcome|surmounted|surpassed|exceeded|transcended)\b',
                    
                    # Emotion-related vocabulary with synonyms
                    r'\b(emotion|emotional|feelings?|mood|mental health|therapy|counseling|support|psychology|psychological|psychiatric|psychiatry|therapeutic|clinical|diagnosis|treatment|healing|recovery|wellness|well-being|mental state|emotional state|psychological state|frame of mind|mindset|attitude|disposition|temperament|character|personality|nature|spirit|soul|psyche|mind|consciousness|awareness|sentience|sensibility|sensitivity|perception|sensation|experience|understanding|comprehension|insight|intuition|empathy|sympathy|compassion|concern|care|regard|consideration|attention|notice|observation|awareness|mindfulness|presence|attentiveness|alertness|vigilance|watchfulness|heedfulness|carefulness|caution|prudence|discretion|judgment|wisdom|understanding|knowledge|comprehension|grasp|appreciation|recognition|realization|acknowledgment|acceptance|approval|endorsement|support|backing|encouragement|reinforcement|strengthening|fortification|consolidation|solidification|stabilization|maintenance|preservation|protection|safeguarding|security|safety|defense|guardianship|custody|care|charge|responsibility|duty|obligation|commitment|dedication|devotion|loyalty|fidelity|faithfulness|trust|confidence|reliance|dependence|interdependence|connection|bond|tie|link|relationship|association|affiliation|alliance|partnership|collaboration|cooperation|coordination|teamwork|unity|solidarity|harmony|accord|agreement|concord|consonance|compatibility|congruence|alignment|attunement|tuning|adjustment|adaptation|accommodation|assimilation|integration|incorporation|inclusion|belonging|membership|participation|involvement|engagement|immersion|absorption|engrossment|fascination|captivation|enchantment|spell|charm|appeal|attraction|allure|magnetism|charisma|presence|aura|vibe|energy|spirit|essence|substance|core|heart|center|nucleus|hub|focus|point|locus|site|location|place|position|situation|circumstance|condition|state|status|standing|position|rank|level|grade|class|category|type|kind|sort|variety|species|genus|family|order|class|phylum|kingdom|domain|realm|sphere|area|field|domain|territory|province|region|zone|sector|division|section|segment|portion|part|component|element|factor|aspect|feature|characteristic|quality|property|attribute|trait|mark|sign|indication|symptom|manifestation|expression|demonstration|display|exhibition|presentation|show|performance|rendition|interpretation|reading|understanding|comprehension|grasp|mastery|command|control|dominion|jurisdiction|authority|power|influence|sway|leverage|impact|effect|consequence|result|outcome|product|yield|return|benefit|advantage|gain|profit|value|worth|merit|virtue|excellence|superiority|supremacy|preeminence|distinction|eminence|prominence|notability|renown|fame|celebrity|stardom|popularity|recognition|reputation|repute|standing|status|position|rank|level|grade|class|category|type|kind|sort|variety|species|genus|family|order|class|phylum|kingdom|domain|realm|sphere|area|field|domain|territory|province|region|zone|sector|division|section|segment|portion|part|component|element|factor|aspect|feature|characteristic|quality|property|attribute|trait|mark|sign|indication|symptom|manifestation|expression|demonstration|display|exhibition|presentation|show|performance|rendition|interpretation|reading|understanding|comprehension|grasp|mastery|command|control|dominion|jurisdiction|authority|power|influence|sway|leverage|impact|effect|consequence|result|outcome|product|yield|return|benefit|advantage|gain|profit|value|worth|merit|virtue|excellence|superiority|supremacy|preeminence|distinction|eminence|prominence|notability|renown|fame|celebrity|stardom|popularity|recognition|reputation|repute)\b',
                    
                    # Intense emotional states
                    r'\b(?:overwhelmed|devastated|shattered|destroyed|broken|crushed|consumed|tormented|plagued|haunted|hounded|besieged|assailed|bombarded|inundated|flooded|deluged|swamped|submerged|engulfed|enveloped|enveloped|enclosed|encircled|surrounded|beset|afflicted|stricken|smitten|seized|gripped|clutched|grasped|held|captured|ensnared|entrapped|imprisoned|confined|restrained|restricted|limited|constrained|bound|tied|shackled|fettered|manacled|handcuffed|chained|roped|leashed|collared|yoked|bridled|muzzled|gagged|silenced|muted|stifled|suppressed|repressed|oppressed|depressed|compressed|condensed|contracted|shrunk|reduced|diminished|lessened|minimized|belittled|devalued|degraded|debased|corrupted|perverted|distorted|twisted|warped|bent|flexed|stretched|strained|stressed|tensioned|tightened|constricted|restricted|limited|confined|imprisoned|enslaved|bound|tied|shackled|fettered|manacled|handcuffed|chained|roped|leashed|collared|yoked|bridled|muzzled|gagged|silenced|muted|stifled|suppressed|repressed|oppressed|depressed|compressed|condensed|contracted|shrunk|reduced|diminished|lessened|minimized|belittled|devalued|degraded|debased|corrupted|perverted|distorted|twisted|warped|bent|flexed|stretched|strained|stressed|tensioned|tightened|constricted|restricted|limited|confined|imprisoned|enslaved|bound|tied|shackled|fettered|manacled|handcuffed|chained|roped|leashed|collared|yoked|bridled|muzzled|gagged|silenced|muted|stifled|suppressed|repressed|oppressed|depressed|compressed|condensed|contracted|shrunk|reduced|diminished|lessened|minimized|belittled|devalued|degraded|debased|corrupted|perverted|distorted|twisted|warped|bent|flexed|stretched|strained|stressed|tensioned|tightened|constricted|restricted|limited|confined|imprisoned|enslaved)\b',
                    
                    # Emotional intensity modifiers
                    r'\b(?:utterly|completely|totally|absolutely|entirely|wholly|fully|thoroughly|entirely|exhaustively|comprehensively|intensively|extensively|profoundly|deeply|intensely|severely|acutely|critically|gravely|seriously|heavily|greatly|significantly|substantially|considerably|markedly|noticeably|remarkably|strikingly|dramatically|radically|fundamentally|profoundly|deeply|intensely|severely|acutely|critically|gravely|seriously|heavily|greatly|significantly|substantially|considerably|markedly|noticeably|remarkably|strikingly|dramatically|radically|fundamentally|profoundly|deeply|intensely|severely|acutely|critically|gravely|seriously|heavily|greatly|significantly|substantially|considerably|markedly|noticeably|remarkably|strikingly|dramatically|radically|fundamentally)\s+(?:sad|depressed|angry|anxious|happy|excited|afraid|terrified|panicked|alarmed|uneasy|tense|stressed|distressed|agitated|restless|jittery|overwhelmed|devastated|shattered|destroyed|broken|crushed|consumed|tormented|plagued|haunted|hounded|besieged|assailed|bombarded|inundated|flooded|deluged|swamped|submerged|engulfed|enveloped|enclosed|encircled|surrounded|beset|afflicted|stricken|smitten|seized|gripped|clutched|grasped|held|captured|ensnared|entrapped|imprisoned|confined|restrained|restricted|limited|constrained|bound|tied|shackled|fettered|manacled|handcuffed|chained|roped|leashed|collared|yoked|bridled|muzzled|gagged|silenced|muted|stifled|suppressed|repressed|oppressed|depressed|compressed|condensed|contracted|shrunk|reduced|diminished|lessened|minimized|belittled|devalued|degraded|debased|corrupted|perverted|distorted|twisted|warped|bent|flexed|stretched|strained|stressed|tensioned|tightened|constricted|restricted|limited|confined|imprisoned|enslaved)\b',
                    
                    # Emotional vocabulary with intensity scales
                    r'\b(?:mild|slight|subtle|gentle|soft|light|faint|weak|moderate|medium|average|intermediate|significant|substantial|considerable|marked|noticeable|remarkable|striking|dramatic|radical|extreme|severe|intense|acute|critical|grave|serious|heavy|great|profound|deep|extreme|severe|intense|acute|critical|grave|serious|heavy|great|profound|deep|extreme|severe|intense|acute|critical|grave|serious|heavy|great|profound|deep)\s+(?:sadness|depression|melancholy|sorrow|grief|anguish|despair|hopelessness|helplessness|worthlessness|emptiness|loneliness|isolation|abandonment|rejection|anger|rage|fury|wrath|irritation|annoyance|vexation|exasperation|aggravation|provocation|agitation|upset|disturbance|perturbation|disquiet|anxiety|worry|nervousness|apprehension|fear|terror|panic|alarm|unease|tension|stress|distress|agitation|restlessness|jitteriness|happiness|joy|elation|euphoria|excitement|enthusiasm|eagerness|anticipation|hope|optimism|confidence|assurance|certainty|security|contentment|satisfaction|pleasure|gratification|fulfillment|relief|comfort|ease|peace|calm|serenity|tranquility|equilibrium|balance|harmony|alignment|attunement)\b',
                    
                    # Emotional triggers and catalysts
                    r'\b(?:trigger|triggered|catalyst|catalyzed|spark|sparked|ignite|ignited|inflame|inflamed|provoke|provoked|incite|incited|instigate|instigated|prompt|prompted|induce|induced|cause|caused|lead|led|result|resulted|effect|affected|impact|impacted|influence|influenced|affect|affected|shape|shaped|form|formed|create|created|produce|produced|generate|generated|bring|brought|give|gave|make|made|render|rendered|reduce|reduced|transform|transformed|change|changed|alter|altered|modify|modified|convert|converted|turn|turned|shift|shifted|switch|switched|transition|transitioned|evolve|evolved|develop|developed|progress|progressed|advance|advanced|move|moved|drive|driven|push|pushed|pull|pulled|draw|drawn|attract|attracted|repel|repelled|compel|compelled|force|forced|coerce|coerced|pressure|pressured|stress|stressed|strain|strained|tension|tensioned|stretch|stretched|compress|compressed|condense|condensed|contract|contracted|expand|expanded|extend|extended|elongate|elongated|lengthen|lengthened|shorten|shortened|shrink|shrank|reduce|reduced|diminish|diminished|lessen|lessened|minimize|minimized|maximize|maximized|amplify|amplified|augment|augmented|increase|increased|decrease|decreased|grow|grew|shrink|shrank|swell|swelled|surge|surged|spike|spiked|plummet|plummeted|plunge|plunged|drop|dropped|fall|fell|rise|rose|soar|soared|skyrocket|skyrocketed|escalate|escalated|intensify|intensified|heighten|heightened|sharpen|sharpened|focus|focused|concentrate|concentrated|center|centered|ground|grounded|anchor|anchored|stabilize|stabilized|balance|balanced|harmonize|harmonized|align|aligned|adjust|adjusted|adapt|adapted|accommodate|accommodated|conform|conformed|reconcile|reconciled|integrate|integrated|incorporate|incorporated|assimilate|assimilated|absorb|absorbed|engulf|engulfed|consume|consumed|devour|devoured|overwhelm|overwhelmed|overpower|overpowered|overcome|overcame|surmount|surmounted|surpass|surpassed|exceed|exceeded|transcend|transcended)\b',
                    
                    # Emotional support and therapy language
                    r'\b(?:therapy|therapeutic|counseling|counselor|therapist|psychologist|psychiatrist|mental health|emotional support|support group|support system|support network|support structure|support mechanism|support framework|support foundation|support pillar|support beam|support column|support bracket|support brace|support prop|support stay|support strut|support tie|support rod|support bar|support rail|support track|support guide|support channel|support groove|support slot|support hole|support cavity|support void|support space|support area|support zone|support region|support sector|support division|support section|support segment|support portion|support part|support component|support element|support factor|support aspect|support feature|support characteristic|support quality|support property|support attribute|support trait|support mark|support sign|support indication|support symptom|support manifestation|support expression|support demonstration|support display|support exhibition|support presentation|support show|support performance|support rendition|support interpretation|support reading|support understanding|support comprehension|support grasp|support mastery|support command|support control|support dominion|support jurisdiction|support authority|support power|support influence|support sway|support leverage|support impact|support effect|support consequence|support result|support outcome|support product|support yield|support return|support benefit|support advantage|support gain|support profit|support value|support worth|support merit|support virtue|support excellence|support superiority|support supremacy|support preeminence|support distinction|support eminence|support prominence|support notability|support renown|support fame|support celebrity|support stardom|support popularity|support recognition|support reputation|support repute)\b',
                    
                    # Emotional validation and acknowledgment
                    r'\b(?:validate|validated|validation|acknowledge|acknowledged|acknowledgment|recognize|recognized|recognition|accept|accepted|acceptance|approve|approved|approval|endorse|endorsed|endorsement|support|supported|support|back|backed|backing|encourage|encouraged|encouragement|reinforce|reinforced|reinforcement|strengthen|strengthened|strengthening|fortify|fortified|fortification|consolidate|consolidated|consolidation|solidify|solidified|solidification|stabilize|stabilized|stabilization|maintain|maintained|maintenance|preserve|preserved|preservation|protect|protected|protection|safeguard|safeguarded|safeguarding|secure|secured|security|shield|shielded|shielding|defend|defended|defense|guard|guarded|guarding|watch|watched|watching|monitor|monitored|monitoring|supervise|supervised|supervision|oversee|oversaw|overseeing|manage|managed|management|control|controlled|control|regulate|regulated|regulation|govern|governed|governing|direct|directed|directing|guide|guided|guiding|lead|led|leading|steer|steered|steering|navigate|navigated|navigating|pilot|piloted|piloting|helm|helmed|helming|command|commanded|commanding|rule|ruled|ruling|dominate|dominated|dominating|master|mastered|mastering|conquer|conquered|conquering|vanquish|vanquished|vanquishing|overcome|overcame|overcoming|surmount|surmounted|surmounting|surpass|surpassed|surpassing|exceed|exceeded|exceeding|transcend|transcended|transcending|ascend|ascended|ascending|soar|soared|soaring|fly|flew|flying|rise|rose|rising|elevate|elevated|elevating|lift|lifted|lifting|raise|raised|raising|boost|boosted|boosting|propel|propelled|propelling|drive|driven|driving|push|pushed|pushing|pull|pulled|pulling|draw|drawn|drawing|attract|attracted|attracting|allure|allured|alluring|entice|enticed|enticing|lure|lured|luring|seduce|seduced|seducing|charm|charmed|charming|enchant|enchanted|enchanting|bewitch|bewitched|bewitching|captivate|captivated|captivating|fascinate|fascinated|fascinating|intrigue|intrigued|intriguing|interest|interested|interesting|engage|engaged|engaging|involve|involved|involving|absorb|absorbed|absorbing|engross|engrossed|engrossing|immerse|immersed|immersing|envelop|enveloped|enveloping|enfold|enfolded|enfolding|embrace|embraced|embracing|hold|held|holding|cradle|cradled|cradling|support|supported|supporting|sustain|sustained|sustaining|maintain|maintained|maintaining|preserve|preserved|preserving|protect|protected|protecting|safeguard|safeguarded|safeguarding|secure|secured|securing|shield|shielded|shielding|defend|defended|defending|guard|guarded|guarding|watch|watched|watching|monitor|monitored|monitoring|supervise|supervised|supervising|oversee|oversaw|overseeing)\b'
                ],
           
                'context': ['emotion', 'feelings', 'mental health', 'therapy', 'counseling', 'support', 'psychology', 'psychological', 'psychiatric', 'psychiatry', 'therapeutic', 'clinical', 'diagnosis', 'treatment', 'healing', 'recovery', 'wellness', 'well-being'],
                'negative_indicators': ['code', 'program', 'function', 'algorithm', 'debug', 'compile', 'run', 'execute', 'syntax', 'variable', 'loop', 'array', 'object', 'class', 'method', 'api', 'database', 'frontend', 'backend', 'full-stack', 'devops', 'git', 'github', 'deployment'],
            },
            AgentType.CODING: {
                'explicit': [
                    r'\b(write|create|implement|build|develop|generate|design|code)\s+(?:a\s+)?(?:python|java|javascript|c\+\+|c#|go|ruby|php|swift|kotlin|rust|typescript|html|css|sql|bash|shell)\s+(?:code|program|function|script|algorithm|class|method|module|package|library|application|app|website|web|api|database)\b',
                    r'\b(?:python|java|javascript|c\+\+|c#|go|ruby|php|swift|kotlin|rust|typescript)\s+(?:code|program|function|script|algorithm|class|method|module|package|library|application|app)\b',
                    r'\bdebug\s+(?:the\s+)?(?:code|program|script|function|algorithm|application)\b',
                    r'\boptimize\s+(?:the\s+)?(?:code|program|function|algorithm|performance)\b',
                    r'\brefactor\s+(?:the\s+)?(?:code|function|class|method)\b',
                    r'\b3d\s+(?:graphics|visualization|rendering|modeling|animation|code|program|function|algorithm)\b',
                    r'\b(web|mobile|desktop|game|ai|machine learning|data science)\s+(?:development|programming|code|application|app)\b',
                    r'\b(api|rest|graphql|database|sql|nosql)\s+(?:development|design|implementation|code)\b',
                    r'\b(import|def|class|function|return|if|else|for|while|try|except|async|await)\s+\w+\b',
                    r'\b\s*{\s*}\s*|\b\s*\(\s*\)\s*|\b\s*\[\s*\]\s*',  # Code structure indicators
                ],
                'context': ['code', 'program', 'function', 'algorithm', 'debug', 'compile', 'run', 'execute', 'syntax', 'variable', 'loop', 'array', 'object', 'class', 'method', 'api', 'database', 'frontend', 'backend', 'full-stack', 'devops', 'git', 'github', 'deployment'],
                'negative_indicators': ['feel', 'emotion', 'sad', 'happy', 'calculate', 'solve', 'equation', 'formula', 'concentration', 'ph']
            },
            
            AgentType.PHYSICS_CHEMISTRY: {
                'explicit': [
                    r'\b(calculate|determine|find|solve|what is|how much|how many|compute|evaluate)\s+(?:the\s+)?(?:concentration|ph|molarity|molality|equilibrium|reaction|yield|rate|velocity|acceleration|force|energy|power|work|momentum|temperature|pressure|volume|density|mass|weight|mole|molar)\b',
                    r'\b(chemical|physics|thermodynamics|mechanics|electromagnetism|quantum|relativity|optics|acoustics|fluid dynamics)\s+(?:problem|question|equation|formula|calculation)\b',
                    r'\b[Hh]2[Oo]|[Cc][Oo]2|[Cc][Hh]4|[Nn][Hh]3|[Nn][Aa][Cc][Ll]|[Cc][Aa][Cc][Oo]3|[Hh]2[Ss][Oo]4|[Cc]6[Hh]12[Oo]6|[Oo]2|[Nn]2|[Hh][Cc][Ll]|[Nn][Aa][Oo][Hh]\b',
                    r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+\b(?=.*\b(?:calculate|determine|find|solve|what is|how much|how many)\b)',
                    r'\b(?:f=ma|e=mc\^2|pv=nrt|v=ir|ke=½mv²|pe=mgh|p=mv|a=v/t|s=vt|ω=θ/t|τ=r×f)\b',
                    r'\b(?:speed of light|planck constant|gravitational constant|boltzmann constant|gas constant)\b',
                    r'\b(?:molecular weight|atomic mass|molar mass|stoichiometry|limiting reagent|titration|buffer|solution)\b',
                ],
                'context': ['physics', 'chemistry', 'science', 'molecule', 'atom', 'element', 'compound', 'reaction', 'equation', 'formula', 'stoichiometry', 'thermodynamics', 'kinetics', 'equilibrium', 'acid', 'base', 'ph', 'concentration', 'molarity', 'molality'],
                'negative_indicators': ['feel', 'emotion', 'code', 'program', 'function', 'algorithm', 'debug']
            },
            
            AgentType.MATH: {
                'explicit': [
                    r'\b(solve|calculate|find|determine|evaluate|compute|simplify|integrate|differentiate|derive|factor|expand|reduce)\s+(?:the\s+)?(?:equation|expression|integral|derivative|limit|sum|product|matrix|vector|polynomial|function|value|problem)\b',
                    r'\b(?:algebra|geometry|calculus|trigonometry|statistics|probability|linear algebra|discrete math|number theory)\s+(?:problem|question|equation|formula|calculation|proof)\b',
                    r'\b(?:x|y|z|a|b|c)\s*=\s*[^=]+\b',
                    r'\b(?:\d+\s*[+\-*/^]\s*\d+|\d+\.\d+\s*[+\-*/^]\s*\d+\.\d+|\d+\s*\^\s*\d+)\b',
                    r'\b(?:sin|cos|tan|log|ln|exp|sqrt|∫|∑|∏|∂|∇|∞|π|e)\b',
                    r'\b(?:matrix|vector|determinant|eigenvalue|eigenvector|linear equation|quadratic equation|cubic equation)\b',
                    r'\b(?:fibonacci|prime|factorial|permutation|combination|binomial|probability|distribution|mean|median|mode|standard deviation)\b',
                ],
                'context': ['math', 'mathematics', 'algebra', 'geometry', 'calculus', 'trigonometry', 'statistics', 'probability', 'equation', 'formula', 'solve', 'calculate', 'integrate', 'differentiate', 'matrix', 'vector', 'function', 'graph', 'plot'],
                'negative_indicators': ['feel', 'emotion', 'code', 'program', 'chemistry', 'physics', 'molecule', 'reaction', 'concentration', 'ph']
            }
        }
        
        # Context clues for disambiguation
        self.context_clues = {
            'coding_indicators': ['python', 'java', 'javascript', 'c++', 'c#', 'code', 'program', 'function', 'algorithm', 'debug', 'compile', 'run', 'syntax', 'variable', 'class', 'method'],
            'science_indicators': ['calculate', 'determine', 'find', 'solve', 'concentration', 'ph', 'molarity', 'molecule', 'atom', 'element', 'compound', 'reaction', 'equation', 'formula', 'physics', 'chemistry'],
            'math_indicators': ['solve', 'calculate', 'equation', 'integral', 'derivative', 'algebra', 'geometry', 'calculus', 'matrix', 'vector', 'function', 'graph'],
            'emotion_indicators': ['feel', 'emotion', 'sad', 'happy', 'angry', 'depressed', 'anxious', 'i am', 'i feel', 'makes me', 'i felt']
        }
        
        # Edge case handlers
        self.edge_case_handlers = {
            'ambiguous_coding_math': self._handle_coding_math_ambiguity,
            'ambiguous_science_math': self._handle_science_math_ambiguity,
            'ambiguous_emotion_general': self._handle_emotion_general_ambiguity,
            'multiple_domains': self._handle_multiple_domains,
            'no_clear_domain': self._handle_no_clear_domain
        }
    
    def detect_agent_enhanced(self, query: str) -> DetectionResult:
        """Enhanced agent detection with comprehensive analysis"""
        text = query.lower().strip()
        
        # Step 1: Calculate confidence scores for each agent type
        scores = {}
        reasons = {}
        
        for agent_type in [AgentType.EMOTION_ANALYSIS, AgentType.CODING, 
                          AgentType.PHYSICS_CHEMISTRY, AgentType.MATH]:
            score, reason = self._calculate_agent_score(text, agent_type)
            scores[agent_type] = score
            reasons[agent_type] = reason
        
        # Step 2: Handle edge cases and ambiguities
        result = self._handle_edge_cases(text, scores, reasons)
        
        # Step 3: Generate context clues
        context_clues = self._extract_context_clues(text)
        
        # Step 4: Build final result
        return DetectionResult(
            agent_type=result['primary_agent'],
            confidence=result['confidence'],
            reasons=result['reasons'],
            alternative_agents=result['alternatives'],
            context_clues=context_clues
        )
    
    def _calculate_agent_score(self, text: str, agent_type: AgentType) -> Tuple[float, List[str]]:
        """Calculate confidence score for a specific agent type"""
        score = 0.0
        reasons = []
        
        patterns = self.patterns[agent_type]
        
        # Check explicit patterns
        for pattern in patterns['explicit']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                score += 2.0 * len(matches)
                reasons.append(f"Matched explicit pattern: {pattern[:50]}...")
        
        # Check context keywords
        context_score = 0
        for keyword in patterns['context']:
            if keyword in text:
                context_score += 1
                reasons.append(f"Found context keyword: {keyword}")
        score += context_score * 0.5
        
        # Check negative indicators
        negative_score = 0
        for indicator in patterns.get('negative_indicators', []):
            if indicator in text:
                negative_score += 1
                reasons.append(f"Found negative indicator: {indicator}")
        score -= negative_score * 1.5
        
        # Apply TextBlob sentiment for emotion detection
        if agent_type == AgentType.EMOTION_ANALYSIS:
            blob = TextBlob(text)
            sentiment_score = abs(blob.sentiment.polarity)
            if sentiment_score > 0.3:
                score += sentiment_score * 2
                reasons.append(f"Strong emotional sentiment: {sentiment_score:.2f}")
        
        return min(score, 10.0), reasons
    
    def _handle_edge_cases(self, text: str, scores: Dict[AgentType, float], reasons: Dict[AgentType, List[str]]) -> Dict:
        """Handle various edge cases and ambiguities"""
        
        # Case 1: No clear winner
        max_score = max(scores.values())
        if max_score < 2.0:
            return {
                'primary_agent': AgentType.GENERAL_CONVERSATION,
                'confidence': 0.3,
                'reasons': ['No clear domain detected'],
                'alternatives': [(agent, score) for agent, score in scores.items() if score > 0]
            }
        
        # Case 2: Close scores between top agents
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0][1] - sorted_scores[1][1] < 1.0:
            return self._handle_close_scores(text, sorted_scores, reasons)
        
        # Case 3: Multiple domains detected
        high_scores = [(agent, score) for agent, score in scores.items() if score > 3.0]
        if len(high_scores) > 1:
            return self._handle_multiple_domains(text, high_scores, reasons)
        
        # Normal case: Clear winner
        primary_agent = max(scores.items(), key=lambda x: x[1])[0]
        return {
            'primary_agent': primary_agent,
            'confidence': min(scores[primary_agent] / 10.0, 1.0),
            'reasons': reasons[primary_agent],
            'alternatives': [(agent, score) for agent, score in sorted_scores[1:3] if score > 0]
        }
    
    def _handle_close_scores(self, text: str, sorted_scores: List[Tuple[AgentType, float]], reasons: Dict) -> Dict:
        """Handle cases where top scores are very close"""
        top_agent, top_score = sorted_scores[0]
        second_agent, second_score = sorted_scores[1]
        
        # Use context clues to break ties
        context_clues = self._extract_context_clues(text)
        
        if top_agent == AgentType.CODING and second_agent == AgentType.MATH:
            if any(indicator in text for indicator in ['python', 'java', 'javascript', 'code', 'function', 'program']):
                return {
                    'primary_agent': AgentType.CODING,
                    'confidence': 0.7,
                    'reasons': ['Coding context indicators found'] + reasons[top_agent],
                    'alternatives': [(second_agent, second_score)]
                }
            else:
                return {
                    'primary_agent': AgentType.MATH,
                    'confidence': 0.7,
                    'reasons': ['Math context indicators found'] + reasons[second_agent],
                    'alternatives': [(top_agent, top_score)]
                }
        
        # Similar logic for other close pairs
        return {
            'primary_agent': top_agent,
            'confidence': 0.6,
            'reasons': ['Close decision based on context'] + reasons[top_agent],
            'alternatives': [(second_agent, second_score)]
        }
    
    def _handle_multiple_domains(self, text: str, high_scores: List[Tuple[AgentType, float]], reasons: Dict) -> Dict:
        """Handle cases where multiple domains are detected"""
        # Prioritize based on specificity
        domain_priority = {
            AgentType.EMOTION_ANALYSIS: 4,
            AgentType.CODING: 3,
            AgentType.PHYSICS_CHEMISTRY: 2,
            AgentType.MATH: 1
        }
        
        best_agent = max(high_scores, key=lambda x: (x[1] + domain_priority[x[0]] * 0.5))
        
        return {
            'primary_agent': best_agent[0],
            'confidence': min(best_agent[1] / 10.0, 0.8),
            'reasons': ['Multiple domains detected, prioritized based on specificity'] + reasons[best_agent[0]],
            'alternatives': [(agent, score) for agent, score in high_scores if agent != best_agent[0]]
        }
    
    def _extract_context_clues(self, text: str) -> List[str]:
        """Extract context clues from the text"""
        clues = []
        
        for category, indicators in self.context_clues.items():
            for indicator in indicators:
                if indicator in text:
                    clues.append(f"{category}: {indicator}")
        
        return clues
    
    def _handle_coding_math_ambiguity(self, text: str) -> Dict:
        """Handle coding vs math ambiguity"""
        coding_indicators = sum(1 for indicator in self.context_clues['coding_indicators'] if indicator in text)
        math_indicators = sum(1 for indicator in self.context_clues['math_indicators'] if indicator in text)
        
        if coding_indicators > math_indicators:
            return {'agent': AgentType.CODING, 'confidence': 0.7}
        else:
            return {'agent': AgentType.MATH, 'confidence': 0.7}
    
    def _handle_science_math_ambiguity(self, text: str) -> Dict:
        """Handle science vs math ambiguity"""
        science_indicators = sum(1 for indicator in self.context_clues['science_indicators'] if indicator in text)
        math_indicators = sum(1 for indicator in self.context_clues['math_indicators'] if indicator in text)
        
        if science_indicators > math_indicators:
            return {'agent': AgentType.PHYSICS_CHEMISTRY, 'confidence': 0.7}
        else:
            return {'agent': AgentType.MATH, 'confidence': 0.7}
    
    def _handle_emotion_general_ambiguity(self, text: str) -> Dict:
        """Handle emotion vs general conversation ambiguity"""
        emotion_indicators = sum(1 for indicator in self.context_clues['emotion_indicators'] if indicator in text)
        
        if emotion_indicators > 2:
            return {'agent': AgentType.EMOTION_ANALYSIS, 'confidence': 0.8}
        else:
            return {'agent': AgentType.GENERAL_CONVERSATION, 'confidence': 0.6}
    
    def _handle_no_clear_domain(self, text: str) -> Dict:
        """Handle cases with no clear domain"""
        return {'agent': AgentType.GENERAL_CONVERSATION, 'confidence': 0.5}

# Global detection function for backward compatibility
def detect_agent(query: str) -> str:
    """Enhanced unified detection function"""
    detector = EnhancedUnifiedAgentDetector()
    result = detector.detect_agent_enhanced(query)
    return result.agent_type.value

# Comprehensive testing function
def test_comprehensive_detection():
    """Test comprehensive detection with edge cases"""
    test_cases = [
        # Emotion cases
        ("I feel really sad and empty inside", AgentType.EMOTION_ANALYSIS),
        ("I am so anxious about everything", AgentType.EMOTION_ANALYSIS),
        ("This makes me feel depressed", AgentType.EMOTION_ANALYSIS),
        
        # Coding cases
        ("Write a Python function to calculate fibonacci numbers", AgentType.CODING),
        ("Create a JavaScript program for 3D graphics rendering", AgentType.CODING),
        ("Implement a C++ class for handling HTTP requests", AgentType.CODING),
        ("Debug this Python code that's giving me errors", AgentType.CODING),
        
        # Science cases
        ("Calculate the pH of a 0.1M HCl solution", AgentType.PHYSICS_CHEMISTRY),
        ("What is the concentration of NaCl in this solution", AgentType.PHYSICS_CHEMISTRY),
        ("Determine the molecular weight of C6H12O6", AgentType.PHYSICS_CHEMISTRY),
        ("Find the velocity of an object under constant acceleration", AgentType.PHYSICS_CHEMISTRY),
        
        # Math cases
        ("Solve the equation x^2 + 3x + 2 = 0", AgentType.MATH),
        ("Calculate the integral of x^2 from 0 to 1", AgentType.MATH),
        ("Find the derivative of sin(x)cos(x)", AgentType.MATH),
        ("Simplify the expression (x+1)(x-1)", AgentType.MATH),
        
        # Edge cases
        ("I feel like I need to solve this math problem", AgentType.MATH),  # Emotion + Math
        ("Write code to calculate the pH of a solution", AgentType.CODING),  # Coding + Science
        ("I am trying to implement a physics simulation in Python", AgentType.CODING),  # Coding + Science
        ("Hello, how are you doing today?", AgentType.GENERAL_CONVERSATION),
        ("Can you help me with something?", AgentType.GENERAL_CONVERSATION),
    ]
    
    detector = EnhancedUnifiedAgentDetector()
    
    print("Testing comprehensive agent detection:")
    print("=" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        result = detector.detect_agent_enhanced(query)
        status = "✓" if result.agent_type == expected else "✗"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected.value}")
        print(f"   Detected: {result.agent_type.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Reasons: {', '.join(result.reasons[:2])}")
        print()
        
        if result.agent_type == expected:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    test_comprehensive_detection()
